import itertools
import pinecone
import pandas as pd

#Connecting to Pinecone Server
api_key = "86378080-7c27-464a-acbb-9e83d7455bcb"

pinecone.init(api_key=api_key, environment='northamerica-northeast1-gcp')

#Connect to your indexes
index_name = "wiesnseb-test"

index = pinecone.Index(index_name=index_name)


# Getting Index Details
#print(pinecone.describe_index(index_name))


data = {
    'ticketno': [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010],
    'complaints': [
        'Broken navigation button on the website',
        'Incorrect pricing displayed for a product',
        'Unable to reset password',
        'App crashes on the latest iOS update',
        'Payment processing error during checkout',
        'Wrong product delivered',
        'Delayed response from customer support',
        'Excessive delivery time for an order',
        'Difficulty in finding a specific product',
        'Error in applying a discount coupon'
    ]
}

df = pd.DataFrame(data)

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("average_word_embeddings_glove.6B.300d")

df["question_vector"] = df.complaints.apply(lambda x: model.encode(str(x)).tolist())

#print(df.head())



def chunks(iterable, batch_size=100):
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))

for batch in chunks([(str(t), v) for t, v in zip(df.ticketno, df.question_vector)]):
    index.upsert(vectors=batch)

#print(index.describe_index_stats())


query_questions = ["bad price"]

query_vectors = [model.encode(str(question)).tolist() for question in query_questions]

query_results = index.query(queries=query_vectors, top_k=5, include_values=False)

print(query_results)