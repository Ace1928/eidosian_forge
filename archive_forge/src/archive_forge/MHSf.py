import logging
import re
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ElasticsearchException
import json


class LogAnalyzer:
    def __init__(self, es_host, es_port, index_name):
        self.es_host = es_host
        self.es_port = es_port
        self.index_name = index_name
        self.connect_to_elasticsearch()
        self.configure_text_processing_tools()
        self.configure_logging()

    def connect_to_elasticsearch(self):
        try:
            self.es = Elasticsearch([{"host": self.es_host, "port": self.es_port}])
            self.logger.info("Successfully connected to Elasticsearch.")
        except ElasticsearchException as e:
            self.logger.error(f"Failed to connect to Elasticsearch: {e}")
            raise

    def configure_text_processing_tools(self):
        self.vectorizer = TfidfVectorizer(stop_words=stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def configure_logging(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(ch)

    def analyze_log(self, log_data):
        try:
            log_text = self.preprocess_log(log_data)
            tokens = self.tokenize_log(log_text)
            vectors = self.vectorizer.fit_transform([" ".join(tokens)])
            clusters = self.cluster_logs(vectors)
            topics = self.model_topics(vectors)
            anomalies = self.detect_anomalies(topics)
            correlations = self.correlate_logs(clusters)
            self.visualize_logs(correlations)
        except Exception as e:
            self.logger.error(f"Error analyzing log: {e}")
            raise

    def preprocess_log(self, log_data):
        log_text = re.sub(r"[\r\n\t]", "", log_data)
        log_text = re.sub(r"\s+", " ", log_text)
        self.logger.debug(f"Preprocessed log: {log_text}")
        return log_text

    def tokenize_log(self, log_text):
        tokens = [self.lemmatizer.lemmatize(word) for word in log_text.split()]
        self.logger.debug(f"Tokenized log: {tokens}")
        return tokens

    def cluster_logs(self, vectors):
        kmeans = KMeans(n_clusters=5)
        clusters = kmeans.fit_predict(vectors)
        self.logger.debug(f"Log clusters: {clusters}")
        return clusters

    def model_topics(self, vectors):
        lda = LatentDirichletAllocation(n_components=5)
        topics = lda.fit_transform(vectors)
        self.logger.debug(f"Log topics: {topics}")
        return topics

    def detect_anomalies(self, topics):
        anomalies = []
        for i, topic in enumerate(topics):
            for j, other_topic in enumerate(topics):
                if i != j:
                    similarity = cosine_similarity([topic], [other_topic])
                    if similarity < 0.5:
                        anomalies.append((i, j))
                        self.logger.debug(
                            f"Anomaly detected between topics {i} and {j}"
                        )
        return anomalies

    def correlate_logs(self, clusters):
        correlations = {}
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if clusters[i] == clusters[j]:
                    correlations.setdefault(i, []).append(j)
                    correlations.setdefault(j, []).append(i)
        self.logger.debug(f"Log correlations: {correlations}")
        return correlations

    def visualize_logs(self, correlations):
        G = nx.Graph()
        for key, values in correlations.items():
            for value in values:
                G.add_edge(key, value)
        plt.figure(figsize=(8, 6))
        nx.draw(
            G,
            with_labels=True,
            node_color="lightblue",
            font_weight="bold",
            node_size=700,
            font_size=10,
        )
        plt.title("Log Correlation Graph")
        plt.show()


if __name__ == "__main__":
    log_analyzer = LogAnalyzer("localhost", 9200, "log_index")
    log_data = json.dumps(
        {
            "message": "Sample log message",
            "level": "INFO",
            "timestamp": "2023-02-20T14:30:00Z",
        }
    )
    log_analyzer.analyze_log(log_data)
