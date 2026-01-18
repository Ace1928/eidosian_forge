from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# News articles
articles = [
    "The government announced a new economic stimulus package to boost the country's economy.",
    "A major technology company unveiled its latest smartphone model at a highly anticipated event.",
    "Scientists discovered a new species of dinosaur in a remote region of South America.",
    "The stock market experienced significant volatility amid concerns over trade tensions.",
    "A renowned artist opened a new exhibition showcasing their latest works.",
]

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(articles)

# Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Print the cluster assignments
for i, label in enumerate(kmeans.labels_):
    print(f"Article {i+1} belongs to Cluster {label+1}")
