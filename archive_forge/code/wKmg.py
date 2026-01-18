"""
This section of the program handles the extraction of metadata from the text data.
"""

import os
import re
from typing import List, Dict


def extract_metadata(file_path: str) -> Dict[str, str]:
    metadata = {}
    with open(file_path, "r") as file:
        content = file.read()
        # Extract title
        title_match = re.search(r"#\s(.+)", content)
        if title_match:
            metadata["title"] = title_match.group(1).strip()
        # Extract author
        author_match = re.search(r"Author:\s(.+)", content)
        if author_match:
            metadata["author"] = author_match.group(1).strip()
        # Extract date
        date_match = re.search(r"Date:\s(.+)", content)
        if date_match:
            metadata["date"] = date_match.group(1).strip()
    return metadata


"""
This section of the program handles chunking and parsing of text data from markdown, text and other text files ensuring all information is extracted verbatim.
"""

import markdown
from bs4 import BeautifulSoup


import os
from typing import List
import markdown
from bs4 import BeautifulSoup
import docx2txt
import pdfplumber
import csv


def extract_text_chunks(file_path: str) -> List[str]:
    _, file_extension = os.path.splitext(file_path)
    text_chunks = []

    try:
        if file_extension == ".md":
            with open(file_path, "r") as file:
                content = file.read()
                html_content = markdown.markdown(content)
                soup = BeautifulSoup(html_content, "html.parser")
                text_chunks = [
                    p.get_text()
                    for p in soup.find_all(
                        ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"]
                    )
                ]
        elif file_extension == ".txt":
            with open(file_path, "r") as file:
                content = file.read()
                text_chunks = content.split("\n\n")
        elif file_extension == ".docx":
            content = docx2txt.process(file_path)
            text_chunks = content.split("\n\n")
        elif file_extension == ".pdf":
            with pdfplumber.open(file_path) as pdf:
                text_chunks = [page.extract_text() for page in pdf.pages]
        elif file_extension == ".csv":
            with open(file_path, "r") as file:
                reader = csv.reader(file)
                text_chunks = ["\n".join(row) for row in reader]
        else:
            print(f"Unsupported file format: {file_extension}")
    except Exception as e:
        print(f"Error processing file: {file_path}")
        print(f"Error message: {str(e)}")

    return text_chunks


def process_files(path: str) -> List[str]:
    all_text_chunks = []

    if os.path.isfile(path):
        all_text_chunks.extend(extract_text_chunks(path))
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                all_text_chunks.extend(extract_text_chunks(file_path))
    else:
        print(f"Invalid path: {path}")

    return all_text_chunks


"""
This section of the program handles the identification of natural language vs code in the text data.
"""

import re


def is_code(text: str) -> bool:
    code_pattern = re.compile(
        r"def |import |class |for |while |if |else |try |except |with |return |yield |@\w+"
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
from langdetect import detect
import logging
import pathlib

# Set up basic logging
logging.basicConfig(level=logging.INFO)


# Function to robustly detect language
def robust_detect_language(text: str) -> str:
    try:
        detected_language = detect(text)
        logging.info(f"Detected Language: {detected_language}")
        return detected_language
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


def cluster_text_chunks(
    text_chunks: List[str], n_clusters: int = 3
) -> Dict[int, List[str]]:
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text_chunks)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)

    # Group text chunks by cluster
    clustered_chunks = {i: [] for i in range(n_clusters)}
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
            token.text for token in doc if not token.is_stop and token.is_alpha
        ]
        keywords.extend(chunk_keywords)
    return keywords


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


def save_analysis_results(file_path: str, analysis_results: Dict[str, Any]):
    with open(file_path, "w") as file:
        json.dump(analysis_results, file, indent=4)


def load_analysis_results(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as file:
        analysis_results = json.load(file)
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


def process_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt;*.md")])
    if file_path:
        # Extract metadata
        metadata = extract_metadata(file_path)

        # Extract text chunks
        text_chunks = extract_text_chunks(file_path)

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
        save_analysis_results("analysis_results.json", analysis_results)

        # Build knowledge graph
        knowledge_graph = build_knowledge_graph(analysis_results)

        # Visualize relationship graph
        visualize_relationship_graph(relationship_graph)

        print("Analysis completed. Results saved to 'analysis_results.json'.")


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
process_button = tk.Button(window, text="Process File", command=process_file)
process_button.pack(pady=10)

visualize_button = tk.Button(
    window, text="Visualize Results", command=visualize_results
)
visualize_button.pack(pady=10)

# Run the main event loop
window.mainloop()
