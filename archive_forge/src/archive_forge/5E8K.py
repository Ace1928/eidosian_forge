"""
This section of the program handles the extraction of metadata from the text data.
"""

import os
import re
import logging
from typing import List, Dict

# Set up basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def extract_metadata(file_path: str) -> Dict[str, str]:
    """
    Extracts metadata from the given file.

    Args:
        file_path (str): Path to the file from which metadata is to be extracted.

    Returns:
        Dict[str, str]: A dictionary containing extracted metadata.
    """
    metadata = {}
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            # Extract title
            title_match = re.search(r"#\s(.+)", content)
            if title_match:
                metadata["title"] = title_match.group(1).strip()
            # Extract author
            author_match = re.search(r"Author:\s(.+)", content, re.IGNORECASE)
            if author_match:
                metadata["author"] = author_match.group(1).strip()
            # Extract date
            date_match = re.search(r"Date:\s(.+)", content, re.IGNORECASE)
            if date_match:
                metadata["date"] = date_match.group(1).strip()
    except Exception as e:
        logging.error(
            f"Error extracting metadata from file {file_path}: {e}", exc_info=True
        )
        raise RuntimeError(
            f"Failed to extract metadata from file {file_path} due to an error: {str(e)}"
        )
    return metadata


"""
This section of the program handles chunking and parsing of text data from various file formats, ensuring all information is extracted verbatim.
"""

import markdown
from bs4 import BeautifulSoup
import docx2txt
import pdfplumber
import csv
import json
import xmltodict


class TextExtractor:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def extract_text_chunks(self) -> List[str]:
        _, file_extension = os.path.splitext(self.file_path)
        text_chunks = []

        try:
            if file_extension == ".md":
                text_chunks = self._extract_md()
            elif file_extension == ".txt":
                text_chunks = self._extract_txt()
            elif file_extension == ".docx":
                text_chunks = self._extract_docx()
            elif file_extension == ".pdf":
                text_chunks = self._extract_pdf()
            elif file_extension == ".csv":
                text_chunks = self._extract_csv()
            elif file_extension == ".json":
                text_chunks = self._extract_json()
            elif file_extension == ".xml":
                text_chunks = self._extract_xml()
            else:
                logging.warning(f"Unsupported file format: {file_extension}")
        except Exception as e:
            logging.error(f"Error processing file: {self.file_path}", exc_info=True)
            raise RuntimeError(
                f"Failed to process file {self.file_path} due to an error: {str(e)}"
            )

        return text_chunks

    def _extract_md(self) -> List[str]:
        with open(self.file_path, "r", encoding="utf-8") as file:
            content = file.read()
            html_content = markdown.markdown(content)
            soup = BeautifulSoup(html_content, "html.parser")
            return [
                p.get_text()
                for p in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"])
            ]

    def _extract_txt(self) -> List[str]:
        with open(self.file_path, "r", encoding="utf-8") as file:
            content = file.read()
            return content.split("\n\n")

    def _extract_docx(self) -> List[str]:
        content = docx2txt.process(self.file_path)
        return content.split("\n\n")

    def _extract_pdf(self) -> List[str]:
        with pdfplumber.open(self.file_path) as pdf:
            return [page.extract_text() for page in pdf.pages]

    def _extract_csv(self) -> List[str]:
        with open(self.file_path, "r", encoding="utf-8") as file:
            reader = csv.reader(file)
            return ["\n".join(row) for row in reader]

    def _extract_json(self) -> List[str]:
        with open(self.file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            return [json.dumps(data, indent=2)]

    def _extract_xml(self) -> List[str]:
        with open(self.file_path, "r", encoding="utf-8") as file:
            data = xmltodict.parse(file.read())
            return [json.dumps(data, indent=2)]


def process_files(path: str) -> List[str]:
    all_text_chunks = []

    if os.path.isfile(path):
        extractor = TextExtractor(path)
        all_text_chunks.extend(extractor.extract_text_chunks())
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                extractor = TextExtractor(file_path)
                all_text_chunks.extend(extractor.extract_text_chunks())
    else:
        logging.error(f"Invalid path: {path}")

    return all_text_chunks


"""
This section of the program handles the identification of natural language vs code in the text data.
"""


def is_code(text: str) -> bool:
    code_pattern = re.compile(
        r"(def |import |class |for |while |if |else |try |except |with |return |yield |@\w+|<\?php|\$\w+|<\?xml|<\w+>|<\/\w+>)"
    )
    return bool(code_pattern.search(text))


def separate_code_and_text(text_chunks: List[str]) -> Dict[str, List[str]]:
    separated_chunks = {"code": [], "text": []}
    for chunk in text_chunks:
        if is_code(chunk):
            separated_chunks["code"].append(chunk)
        else:
            separated_chunks["text"].append(chunk)
    return separated_chunks


"""
This Section of the program handles automatic language detection and translation of natural language text.
"""

import argostranslate.package, argostranslate.translate
from langdetect import detect_langs
import pathlib


# Function to robustly detect language
def robust_detect_language(text: str) -> str:
    try:
        detected_languages = detect_langs(text)
        top_language = max(detected_languages, key=lambda lang: lang.prob)
        logging.info(f"Detected Language: {top_language.lang}")
        return top_language.lang
    except Exception as e:
        logging.error(f"Language detection failed: {e}")
        return None


# Ensure the package index is up-to-date and get available packages
def update_and_load_packages():
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    installed_languages = argostranslate.translate.get_installed_languages()
    language_codes = {lang.code: lang for lang in installed_languages}
    return available_packages, installed_languages, language_codes


# Download and verify translation packages
def download_and_verify_package(source_lang_code: str, target_lang_code: str) -> bool:
    available_packages, _, _ = update_and_load_packages()
    desired_package = next(
        (
            pkg
            for pkg in available_packages
            if pkg.from_code == source_lang_code and pkg.to_code == target_lang_code
        ),
        None,
    )
    if desired_package:
        download_path = desired_package.download()
        argostranslate.package.install_from_path(pathlib.Path(download_path))
        logging.info(f"Package downloaded and installed from {download_path}")
        return True
    else:
        logging.error(
            f"No available package from {source_lang_code} to {target_lang_code}"
        )
        return False


# Enhanced language detection and translation
def translate_text(text: str, target_lang_code="en"):
    detected_language = robust_detect_language(text)
    if detected_language:
        _, installed_languages, language_codes = update_and_load_packages()
        if detected_language not in language_codes:
            if not download_and_verify_package(detected_language, target_lang_code):
                logging.error(
                    f"No available translation package from {detected_language} to {target_lang_code}."
                )
                return text
            # Update language codes after downloading new package
            _, _, language_codes = update_and_load_packages()
        translation = language_codes[detected_language].get_translation(
            language_codes[target_lang_code]
        )
        translated_text = translation.translate(text)
        logging.info(f"Original Text: {text}")
        logging.info(f"Translated Text: {translated_text}")
        return translated_text
    return text


def translate_text_chunks(text_chunks: List[str], target_lang_code="en") -> List[str]:
    translated_chunks = []
    for chunk in text_chunks:
        translated_chunk = translate_text(chunk, target_lang_code)
        translated_chunks.append(translated_chunk)
    return translated_chunks


"""
This section of the program handles clustering of all text data (natural language) after translation (if required).
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def cluster_text_chunks(
    text_chunks: List[str], min_clusters: int = 2, max_clusters: int = 10
) -> Dict[int, List[str]]:
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text_chunks)

    # Determine the optimal number of clusters using silhouette score
    best_score = -1
    best_n_clusters = min_clusters
    for n_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters

    # Perform K-means clustering with the optimal number of clusters
    kmeans = KMeans(n_clusters=best_n_clusters, random_state=42)
    kmeans.fit(X)

    # Group text chunks by cluster
    clustered_chunks = {i: [] for i in range(best_n_clusters)}
    for i, label in enumerate(kmeans.labels_):
        clustered_chunks[label].append(text_chunks[i])

    return clustered_chunks


"""
This section of the program handles topic discovery within the text data.
"""

from gensim import corpora, models


def discover_topics(text_chunks: List[str], num_topics: int = 3) -> List[str]:
    # Tokenize the text chunks
    tokenized_chunks = [chunk.lower().split() for chunk in text_chunks]

    # Create a dictionary and corpus
    dictionary = corpora.Dictionary(tokenized_chunks)
    corpus = [dictionary.doc2bow(chunk) for chunk in tokenized_chunks]

    # Train an LDA model
    lda_model = models.LdaModel(
        corpus, num_topics=num_topics, id2word=dictionary, passes=10
    )

    # Extract discovered topics
    topics = [topic for _, topic in lda_model.print_topics()]

    return topics


"""
This section of the program handles named entity recognition and keyword extraction.
"""

import spacy

# Load the pre-trained English model
nlp = spacy.load("en_core_web_md")


def extract_named_entities(text_chunks: List[str]) -> List[Dict[str, str]]:
    named_entities = []
    for chunk in text_chunks:
        doc = nlp(chunk)
        chunk_entities = [
            {"text": entity.text, "label": entity.label_} for entity in doc.ents
        ]
        named_entities.extend(chunk_entities)
    return named_entities


def extract_keywords(text_chunks: List[str]) -> List[str]:
    keywords = []
    for chunk in text_chunks:
        doc = nlp(chunk)
        chunk_keywords = [
            token.text.lower()
            for token in doc
            if not token.is_stop and token.is_alpha and token.pos_ in ["NOUN", "PROPN"]
        ]
        keywords.extend(chunk_keywords)
    return list(set(keywords))


"""
This section of the program handles sentiment analysis of text data.
"""

from textblob import TextBlob


def analyze_sentiment(text_chunks: List[str]) -> List[Dict[str, float]]:
    sentiments = []
    for chunk in text_chunks:
        blob = TextBlob(chunk)
        sentiment = {
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity,
        }
        sentiments.append(sentiment)
    return sentiments


"""
This section of the program handles cosine similarity calculation between text data.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_cosine_similarity(text_chunks: List[str]) -> List[List[float]]:
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text_chunks)

    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(X)

    return similarity_matrix.tolist()


"""
This section of the program handles analysis and construction of relationships and dependencies between text data.
"""

import networkx as nx


def build_text_relationship_graph(
    text_chunks: List[str], similarity_threshold: float = 0.5
) -> nx.Graph:
    # Calculate cosine similarity between text chunks
    similarity_matrix = calculate_cosine_similarity(text_chunks)

    # Create a graph
    G = nx.Graph()

    # Add nodes and edges based on similarity
    for i in range(len(text_chunks)):
        G.add_node(i, text=text_chunks[i])
        for j in range(i + 1, len(text_chunks)):
            if similarity_matrix[i][j] >= similarity_threshold:
                G.add_edge(i, j, weight=similarity_matrix[i][j])

    return G


"""
This section of the program handles aggregation and comparison of all the extracted information and analysis.
"""

from typing import Any


def aggregate_analysis_results(
    metadata: Dict[str, str],
    separated_chunks: Dict[str, List[str]],
    clustered_chunks: Dict[int, List[str]],
    topics: List[str],
    named_entities: List[Dict[str, str]],
    keywords: List[str],
    sentiments: List[Dict[str, float]],
    similarity_matrix: List[List[float]],
    relationship_graph: nx.Graph,
) -> Dict[str, Any]:
    aggregated_results = {
        "metadata": metadata,
        "separated_chunks": separated_chunks,
        "clustered_chunks": clustered_chunks,
        "topics": topics,
        "named_entities": named_entities,
        "keywords": keywords,
        "sentiments": sentiments,
        "similarity_matrix": similarity_matrix,
        "relationship_graph": relationship_graph,
    }
    return aggregated_results


"""
This section of the program handles the storage and retrieval of the analysis results.
"""

import json
import os
from typing import Dict, Any
import networkx as nx
from networkx.readwrite import json_graph


def save_analysis_results(file_path: str, analysis_results: Dict[str, Any]) -> None:
    """
    Robustly saves the analysis results to a JSON file, ensuring all components including complex data structures
    are serialized correctly.

    Args:
        file_path (str): The path to the file where results will be saved.
        analysis_results (Dict[str, Any]): The analysis results to save, including complex data types.
    """

    def default(obj):
        """Custom serialization for complex objects."""
        if isinstance(obj, nx.Graph):
            return json_graph.node_link_data(
                obj
            )  # Convert Graph to a serializable format
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure directory exists
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(analysis_results, file, indent=4, default=default)


def load_analysis_results(file_path: str) -> Dict[str, Any]:
    """
    Robustly loads the analysis results from a JSON file, ensuring all data including complex structures
    are deserialized correctly.

    Args:
        file_path (str): The path to the file from which results will be loaded.

    Returns:
        Dict[str, Any]: The analysis results including complex data types.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No file found at the specified path: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        analysis_results = json.load(file)

    # Reconstruct complex data types like Graphs
    if "relationship_graph" in analysis_results and isinstance(
        analysis_results["relationship_graph"], dict
    ):
        analysis_results["relationship_graph"] = json_graph.node_link_graph(
            analysis_results["relationship_graph"]
        )

    return analysis_results


"""
This section of the program handles the construction of a knowledge graph based on the analysis results.
"""

import rdflib
from rdflib import Namespace, URIRef, Literal


def build_knowledge_graph(analysis_results: Dict[str, Any]) -> rdflib.Graph:
    # Create a new graph
    graph = rdflib.Graph()

    # Define namespaces
    ex = Namespace("http://example.com/")
    graph.bind("ex", ex)

    # Add metadata to the graph
    metadata_node = URIRef(ex["metadata"])
    for key, value in analysis_results["metadata"].items():
        graph.add((metadata_node, URIRef(ex[key]), Literal(value)))

    # Add topics to the graph
    for i, topic in enumerate(analysis_results["topics"]):
        topic_node = URIRef(ex[f"topic_{i}"])
        graph.add((topic_node, rdflib.RDF.type, URIRef(ex["Topic"])))
        graph.add((topic_node, URIRef(ex["description"]), Literal(topic)))

    # Add named entities to the graph
    for entity in analysis_results["named_entities"]:
        entity_node = URIRef(ex[f"entity_{entity['text']}"])
        graph.add((entity_node, rdflib.RDF.type, URIRef(ex["NamedEntity"])))
        graph.add((entity_node, URIRef(ex["text"]), Literal(entity["text"])))
        graph.add((entity_node, URIRef(ex["label"]), Literal(entity["label"])))

    # Add keywords to the graph
    for keyword in analysis_results["keywords"]:
        keyword_node = URIRef(ex[f"keyword_{keyword}"])
        graph.add((keyword_node, rdflib.RDF.type, URIRef(ex["Keyword"])))
        graph.add((keyword_node, URIRef(ex["text"]), Literal(keyword)))

    return graph


"""
This section of the program handles the visualization of the analysis results.
"""

import matplotlib.pyplot as plt
import networkx as nx


def visualize_relationship_graph(relationship_graph: nx.Graph):
    pos = nx.spring_layout(relationship_graph)
    nx.draw(
        relationship_graph,
        pos,
        with_labels=True,
        node_size=500,
        font_size=10,
        edge_color="gray",
    )
    labels = nx.get_edge_attributes(relationship_graph, "weight")
    nx.draw_networkx_edge_labels(relationship_graph, pos, edge_labels=labels)
    plt.axis("off")
    plt.show()


"""
This section of the program provides a simple user interface to tie all the functionality together.
"""

import tkinter as tk
from tkinter import filedialog
from typing import Dict, Any

import os
import logging


def process_directory():
    """
    Process all files within a selected directory.
    """
    path = filedialog.askdirectory()
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    process_single_file(file_path)
                except Exception as e:
                    logging.error(f"Error processing file {file_path}: {str(e)}")
                    continue


def process_single_file(file_path: str):
    """
    Process a single file by performing a series of analysis tasks and saving the results.

    Args:
        file_path (str): The path to the file to be processed.
    """
    try:
        # Extract metadata
        metadata = extract_metadata(file_path)

        # Extract text chunks
        text_chunks = TextExtractor(file_path).extract_text_chunks()

        # Separate code and text
        separated_chunks = separate_code_and_text(text_chunks)

        # Translate text chunks
        translated_chunks = translate_text_chunks(separated_chunks["text"])

        # Cluster text chunks
        clustered_chunks = cluster_text_chunks(translated_chunks)

        # Discover topics
        topics = discover_topics(translated_chunks)

        # Extract named entities
        named_entities = extract_named_entities(translated_chunks)

        # Extract keywords
        keywords = extract_keywords(translated_chunks)

        # Analyze sentiment
        sentiments = analyze_sentiment(translated_chunks)

        # Calculate cosine similarity
        similarity_matrix = calculate_cosine_similarity(translated_chunks)

        # Build text relationship graph
        relationship_graph = build_text_relationship_graph(translated_chunks)

        # Aggregate analysis results
        analysis_results = aggregate_analysis_results(
            metadata,
            separated_chunks,
            clustered_chunks,
            topics,
            named_entities,
            keywords,
            sentiments,
            similarity_matrix,
            relationship_graph,
        )

        # Save analysis results
        save_analysis_results(f"{file_path}_analysis_results.json", analysis_results)

        # Build knowledge graph
        knowledge_graph = build_knowledge_graph(analysis_results)

        # Visualize relationship graph
        visualize_relationship_graph(relationship_graph)

        print(
            f"Analysis completed for {file_path}. Results saved to '{file_path}_analysis_results.json'."
        )
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {str(e)}")
        raise


def visualize_results():
    file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
    if file_path:
        # Load analysis results
        analysis_results = load_analysis_results(file_path)

        # Visualize relationship graph
        relationship_graph = analysis_results["relationship_graph"]
        visualize_relationship_graph(relationship_graph)


# Create the main window
window = tk.Tk()
window.title("Text Analysis Tool")

# Create buttons
process_button = tk.Button(window, text="Process Directory", command=process_directory)
process_button.pack(pady=10)

visualize_button = tk.Button(
    window, text="Visualize Results", command=visualize_results
)
visualize_button.pack(pady=10)

# Run the main event loop
window.mainloop()
