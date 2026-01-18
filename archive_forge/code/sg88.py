import networkx as nx
import matplotlib.pyplot as plt
from neo4j import GraphDatabase
from typing import List, Dict


class KnowledgeGraph:
    """
    Constructs and manages a dynamic, interactive knowledge graph based on embeddings.
    This module uses NetworkX for in-memory graph operations and Neo4j for persistent graph storage.
    """

    def __init__(self, uri: str, user: str, password: str):
        """
        Initializes the knowledge graph module with a connection to the graph database.
        :param uri: str - URI for the Neo4j database.
        :param user: str - Username for the Neo4j database.
        :param password: str - Password for the Neo4j database.
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.graph = nx.DiGraph()

    def update_graph(self, analysis_results: List[Dict[str, any]]) -> None:
        """
        Updates the knowledge graph with new analysis results, integrating data into Neo4j and NetworkX graph.
        :param analysis_results: List[Dict[str, any]] - Results from data analysis to be integrated into the graph.
        """
        with self.driver.session() as session:
            for result in analysis_results:
                session.run(
                    "MERGE (a:Document {file_path: $file_path}) "
                    "ON CREATE SET a.embedding = $embedding",
                    file_path=result["file_path"],
                    embedding=result["embedding"],
                )
                for relation in result["relations"]:
                    session.run(
                        "MATCH (a:Document {file_path: $file_path}) "
                        "MERGE (b:Document {file_path: $related_file}) "
                        "MERGE (a)-[:RELATED_TO {type: $relation_type}]->(b)",
                        file_path=result["file_path"],
                        related_file=relation["file_path"],
                        relation_type=relation["type"],
                    )
                self.graph.add_node(result["file_path"], embedding=result["embedding"])
                for relation in result["relations"]:
                    self.graph.add_edge(
                        result["file_path"],
                        relation["file_path"],
                        type=relation["type"],
                    )

    def display_graph(self) -> None:
        """
        Renders the interactive knowledge graph using matplotlib for visualization.
        """
        pos = nx.spring_layout(self.graph, seed=42)  # for consistent layout
        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            node_color="skyblue",
            edge_color="k",
            node_size=700,
            font_size=10,
        )
        plt.show()

    def __del__(self):
        """
        Closes the database connection when the object is deleted.
        """
        self.driver.close()
