import os
import hashlib
import ast
import json
import logging
import sqlite3
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from scipy.stats import entropy
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel, Canvas
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Configure logging with a detailed format specification
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

from typing import Optional, Type
import sqlite3


class SQLiteConnectionManager:
    """
    Manages the lifecycle of SQLite database connections, ensuring robust, reusable, and efficient database connectivity.
    This class encapsulates connection handling and transaction management, serving as a foundational component for database interactions.
    It adheres to the context management protocol to facilitate safe use of resources.
    """

    def __init__(self, database_path: str):
        """
        Initializes an SQLiteConnectionManager instance with a specified path to the SQLite database.

        Parameters:
        - database_path (str): The filesystem path to the SQLite database file.
        """
        self.database_path: str = database_path
        self.connection: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None

    def __enter__(self) -> "SQLiteConnectionManager":
        """
        Establishes a database connection upon entering the runtime context, preparing the manager for database operations.

        Returns:
        - SQLiteConnectionManager: The instance with an active database connection.
        """
        self.connection = sqlite3.connect(self.database_path)
        self.cursor = self.connection.cursor()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Type[BaseException]],
    ) -> None:
        """
        Terminates the database connection upon exiting the runtime context, ensuring clean disconnection and handling exceptions gracefully.

        Parameters:
        - exc_type (Optional[Type[BaseException]]): The type of the exception, if any.
        - exc_val (Optional[BaseException]): The exception instance, if any.
        - exc_tb (Optional[Type[BaseException]]): The traceback object, if any.
        """
        if self.connection:
            if exc_type is not None:
                self.connection.rollback()
            else:
                self.connection.commit()
            self.connection.close()

    def commit(self) -> None:
        """
        Commits the current transaction to the database, ensuring that all changes made during the transaction are persisted.
        This method is critical for maintaining data integrity and consistency within the database.
        """
        if self.connection:
            self.connection.commit()


class EmbeddingDatabaseManager:
    """
    Manages the database operations specifically for embeddings, including initialization, insertion, retrieval, and closure.
    """

    def __init__(self, connection_manager: SQLiteConnectionManager):
        """
        Initializes the EmbeddingDatabaseManager with a connection manager to handle database operations.

        Parameters:
        - connection_manager (SQLiteConnectionManager): The connection manager to handle database interactions.
        """
        self.connection_manager = connection_manager

    def initialize_database(self) -> None:
        """
        Initialize the database by creating the necessary table if it does not exist.
        """
        create_table_query = """
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY,
                file_hash TEXT UNIQUE,
                imports TEXT,
                function TEXT,
                docstring TEXT,
                embedding BLOB
            )
        """
        self.connection_manager.cursor.execute(create_table_query)
        self.connection_manager.commit()

    def insert_embedding(
        self,
        file_hash: str,
        imports: str,
        function: str,
        docstring: str,
        embedding: bytes,
    ) -> None:
        """
        Insert an embedding into the database.

        Parameters:
        - file_hash: str - The hash of the file.
        - imports: str - The imports used in the file.
        - function: str - The function code.
        - docstring: str - The docstring of the function.
        - embedding: bytes - The serialized embedding.
        """
        insert_query = """
            INSERT INTO embeddings (file_hash, imports, function, docstring, embedding) 
            VALUES (?, ?, ?, ?, ?)
        """
        try:
            self.connection_manager.cursor.execute(
                insert_query, (file_hash, imports, function, docstring, embedding)
            )
        except sqlite3.DatabaseError as e:
            logging.error(f"Error inserting embedding: {e}")
            raise

    def is_file_processed(self, file_hash: str) -> bool:
        """
        Check if a file has already been processed by querying its hash.

        Parameters:
        - file_hash: str - The hash of the file to check.
        :return: bool - True if the file has been processed, False otherwise.
        """
        check_query = "SELECT id FROM embeddings WHERE file_hash = ?"
        try:
            self.connection_manager.cursor.execute(check_query, (file_hash,))
            return self.connection_manager.cursor.fetchone() is not None
        except sqlite3.DatabaseError as e:
            logging.error(f"Error checking processed file: {e}")
            raise

    def retrieve_embeddings(
        self, search_term: str
    ) -> list[tuple[str, str, str, bytes]]:
        """
        Retrieve embeddings that match the search term either in function or docstring.

        Parameters:
        - search_term: str - The term to search for in function names or docstrings.
        :return: list[tuple[str, str, str, bytes]] - A list of tuples containing imports, function, docstring, and embedding.
        """
        retrieve_query = """
            SELECT imports, function, docstring, embedding 
            FROM embeddings 
            WHERE function LIKE ? OR docstring LIKE ? COLLATE NOCASE
        """
        try:
            self.connection_manager.cursor.execute(
                retrieve_query, ("%" + search_term + "%", "%" + search_term + "%")
            )
            return self.connection_manager.cursor.fetchall()
        except sqlite3.DatabaseError as e:
            logging.error(f"Error retrieving embeddings: {e}")
            raise

    def retrieve_all_embeddings(self) -> list[tuple[str, str, str, bytes]]:
        """
        Retrieve all embeddings from the database using pagination to handle large datasets efficiently.

        :return: list[tuple[str, str, str, bytes]] - A list of tuples containing imports, function, docstring, and embedding.
        """
        retrieve_all_query = (
            "SELECT imports, function, docstring, embedding FROM embeddings"
        )
        try:
            self.connection_manager.cursor.execute(retrieve_all_query)
            results = []
            while True:
                page = self.connection_manager.cursor.fetchmany(
                    1000
                )  # Fetch results in pages of 1000
                if not page:
                    break
                results.extend(page)
            return results
        except sqlite3.DatabaseError as e:
            logging.error(f"Error retrieving all embeddings: {e}")
            raise


class FileManager:
    """
    Manages file operations related to processed and embedded files, including loading, saving, and processing files.
    """

    def __init__(self, processed_files_path: str, embedded_files_path: str):
        """
        Initializes the FileManager with paths to processed and embedded files.

        :param processed_files_path: str - Path to the JSON file containing processed files metadata.
        :param embedded_files_path: str - Path to the JSON file containing embedded files metadata.
        """
        self.processed_files_path = processed_files_path
        self.embedded_files_path = embedded_files_path
        self.processed_files = self._load_json(self.processed_files_path)
        self.embedded_files = self._load_json(self.embedded_files_path)

    def _load_json(self, path: str) -> dict:
        """
        Loads a JSON file from the specified path.

        :param path: str - Path to the JSON file.
        :return: dict - The loaded JSON object, or an empty dictionary if the file does not exist.
        """
        try:
            with open(path, "r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            logging.warning(f"File not found: {path}")
            return {}

    def save_json(self, data: dict, path: str) -> None:
        """
        Saves a dictionary as a JSON file to the specified path.

        :param data: dict - The data to save.
        :param path: str - Path to save the JSON file.
        """
        try:
            with open(path, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=4)
        except IOError as e:
            logging.error(f"Error saving JSON to {path}: {e}")
            raise

    def process_folder(self, folder_path: str, callback) -> Tuple[int, int]:
        """
        Processes all Python files in a specified folder using a callback function.

        :param folder_path: str - Path to the folder containing Python files.
        :param callback: Callable - A callback function to process each Python file.
        :return: Tuple[int, int] - Total number of Python files and number of processed files.
        """
        total_files = 0
        processed_files = 0
        for root, dirs, files in os.walk(folder_path):
            python_files = [file for file in files if file.endswith(".py")]
            total_files += len(python_files)
            for file in python_files:
                file_path = os.path.join(root, file)
                callback(file_path)
                processed_files += 1
        return total_files, processed_files

    def process_file(self, file_path: str, model, db_manager) -> None:
        """
        Processes a single Python file, extracting functions and imports, and storing embeddings in a database.

        :param file_path: str - Path to the Python file.
        :param model: SentenceTransformer - The model used to generate embeddings.
        :param db_manager: EmbeddingDatabaseManager - The database manager to store embeddings.
        """
        try:
            with open(file_path, "r") as file:
                content = file.read()
            file_hash = hashlib.sha256(content.encode()).hexdigest()
            if not db_manager.is_file_processed(file_hash):
                functions, imports, classes = CodeParser.extract_functions_and_imports(
                    content
                )
                embeddings = [
                    model.encode(func["function"], convert_to_tensor=True).numpy()
                    for func in functions
                ]
                for func, embed in zip(functions, embeddings):
                    db_manager.insert_embedding(
                        file_hash,
                        imports,
                        func["function"],
                        func["docstring"],
                        embed.tobytes(),
                    )
        except IOError as e:
            logging.error(f"Error processing file {file_path}: {e}")
            raise


class CodeParser:
    """
    A class dedicated to parsing Python code to extract functions, imports, and classes.
    """

    @staticmethod
    def parse_python_content(content: str) -> ast.Module:
        """
        Parses the content of a Python file into an Abstract Syntax Tree (AST).

        :param content: str - The content of the Python file.
        :return: ast.Module - The AST representation of the content.
        """
        return ast.parse(content)

    @staticmethod
    def extract_functions_from_ast(tree: ast.Module) -> List[Dict[str, str]]:
        """
        Extracts functions from the AST of a Python file.

        :param tree: ast.Module - The AST of the Python file.
        :return: List[Dict[str, str]] - A list of dictionaries containing function code and docstrings.
        """
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                docstring = ast.get_docstring(node) or ""
                function_code = ast.unparse(node)
                functions.append({"function": function_code, "docstring": docstring})
        return functions

    @staticmethod
    def extract_import_statements_from_ast(tree: ast.Module) -> str:
        """
        Extracts all import statements from the AST of a Python file.

        :param tree: ast.Module - The AST of the Python file.
        :return: str - A single string containing all import statements.
        """
        imports = " ".join(
            ast.unparse(node)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
        )
        return imports

    @staticmethod
    def extract_class_definitions_from_ast(tree: ast.Module) -> List[str]:
        """
        Extracts all class definitions from the AST of a Python file.

        :param tree: ast.Module - The AST of the Python file.
        :return: List[str] - A list of strings, each representing a class definition.
        """
        classes = [
            ast.unparse(node)
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
        ]
        return classes

    @staticmethod
    def extract_functions_imports_classes_from_content(
        content: str,
    ) -> Tuple[List[Dict[str, str]], str, List[str]]:
        """
        Extracts functions, imports, and classes from the content of a Python file.

        :param content: str - The content of the Python file.
        :return: Tuple[List[Dict[str, str]], str, List[str]] - A tuple containing a list of functions, a string of imports, and a list of classes.
        """
        tree = CodeParser.parse_python_content(content)
        functions = CodeParser.extract_functions_from_ast(tree)
        imports = CodeParser.extract_import_statements_from_ast(tree)
        classes = CodeParser.extract_class_definitions_from_ast(tree)
        return functions, imports, classes


class GUI:
    """
    Manages the application's main window and interactions, encapsulating all related functionalities.
    """

    def __init__(
        self,
        root: tk.Tk,
        db_manager: EmbeddingDatabaseManager,
        data_analysis_manager: DataAnalysisManager,
        model: SentenceTransformer,
        file_manager: FileManager,
        visualization_manager: DataVisualizationManager,
        advanced_visualization_manager: DataVisualizationManager,
    ):
        """
        Initializes the GUI with all necessary components and managers.
        """
        self.root = root
        self.db_manager = db_manager
        self.data_analysis_manager = data_analysis_manager
        self.model = model
        self.file_manager = file_manager
        self.visualization_manager = visualization_manager
        self.advanced_visualization_manager = advanced_visualization_manager
        self._initialize_user_interface()

    def _initialize_user_interface(self) -> None:
        """
        Configures the user interface, creating and organizing widgets.
        """
        self.frame = tk.Frame(self.root)
        self.frame.pack(padx=10, pady=10)

        self.input_button = tk.Button(
            self.frame,
            text="Select Python Files Folder",
            command=self._handle_folder_selection,
        )
        self.input_button.pack(fill=tk.X)

        self.search_frame = tk.Frame(self.frame)
        self.search_frame.pack(fill=tk.X)

        self.search_entry = tk.Entry(self.search_frame, width=50)
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.search_entry.insert(0, "Enter search term...")

        self.search_button = tk.Button(
            self.search_frame,
            text="Search Embeddings",
            command=self._execute_embedding_search,
        )
        self.search_button.pack(side=tk.LEFT)

        self.results_text = tk.Text(self.frame, height=15, width=80)
        self.results_text.pack(fill=tk.X)

        self.analyze_button = tk.Button(
            self.frame,
            text="Analyze Embeddings",
            command=self._perform_embedding_analysis,
        )
        self.analyze_button.pack(fill=tk.X)

        self.visualize_button = tk.Button(
            self.frame,
            text="Visualize Embeddings",
            command=self._execute_embedding_visualization,
        )
        self.visualize_button.pack(fill=tk.X)

        self.advanced_visualize_button = tk.Button(
            self.frame,
            text="Visualize Analysis Data",
            command=self.advanced_visualization_manager.visualize_advanced_data,
        )
        self.advanced_visualize_button.pack(fill=tk.X)

    def _handle_folder_selection(self) -> None:
        """
        Manages folder selection and initiates processing of files within the selected folder.
        """
        folder_path = filedialog.askdirectory()
        if folder_path:
            total_files, processed_files = self.file_manager.process_folder(
                folder_path,
                lambda fp: self.file_manager.process_file(
                    fp, self.model, self.db_manager
                ),
            )
            self.db_manager.commit_changes()
            tk.messagebox.showinfo(
                "Processing Complete",
                f"All files have been processed. {processed_files}/{total_files} files processed.",
            )
        else:
            tk.messagebox.showinfo(
                "No folder selected", "Please select a valid folder."
            )

    def _execute_embedding_search(self) -> None:
        """
        Executes a search for embeddings based on the user's input and displays the results.
        """
        search_term = self.search_entry.get()
        results = self.db_manager.fetch_embeddings(search_term)
        self._present_search_results(results)

    def _present_search_results(
        self, results: List[Tuple[str, str, str, bytes]]
    ) -> None:
        """
        Presents search results in the text widget.
        """
        self.results_text.delete("1.0", tk.END)
        for imports, function, docstring, _ in results:
            result_text = f"Imports:\n{imports}\n\nFunction:\n{function}\n\nDocstring:\n{docstring}\n\n---\n\n"
            self.results_text.insert(tk.END, result_text)
        logging.info("Search results displayed.")

    def _perform_embedding_analysis(self) -> None:
        """
        Analyzes embeddings using various clustering and dimensionality reduction techniques.
        """
        try:
            results = self.db_manager.fetch_all_embeddings()
            embeddings = [
                np.frombuffer(result[3], dtype=np.float32) for result in results
            ]
            if embeddings:
                matrix = np.stack(embeddings)
                n_clusters = min(5, len(results))
                kmeans_labels, centroids = (
                    self.data_analysis_manager.perform_kmeans_clustering(
                        matrix, n_clusters
                    )
                )
                agglomerative_labels = (
                    self.data_analysis_manager.perform_agglomerative_clustering(
                        matrix, n_clusters
                    )
                )
                dbscan_labels = self.data_analysis_manager.perform_dbscan_clustering(
                    matrix, eps=0.5, min_samples=2
                )
                similarity_matrix = (
                    self.data_analysis_manager.perform_cosine_similarity(matrix)
                )
                pca_embeddings = self.data_analysis_manager.perform_pca(
                    matrix, n_components=2
                )
                tsne_embeddings = self.data_analysis_manager.perform_tsne(
                    matrix, n_components=2
                )
                pairwise_distances = (
                    self.data_analysis_manager.calculate_pairwise_distances(matrix)
                )
                kdtree = self.data_analysis_manager.build_kdtree(matrix)
                entropy_value = self.data_analysis_manager.calculate_entropy(
                    kmeans_labels
                )

                self.data_analysis_manager.store_analysis_results(
                    kmeans_labels,
                    centroids,
                    agglomerative_labels,
                    dbscan_labels,
                    similarity_matrix,
                    pca_embeddings,
                    tsne_embeddings,
                    pairwise_distances,
                    kdtree,
                    entropy_value,
                )

                messagebox.showinfo(
                    "Analysis Complete", "Embedding analysis completed successfully."
                )
            else:
                messagebox.showinfo("No Embeddings", "No embeddings found to analyze.")
        except Exception as e:
            logging.error(f"Error analyzing embeddings: {e}")
            messagebox.showerror(
                "Analysis Error", f"An error occurred during analysis: {str(e)}"
            )

    def _execute_embedding_visualization(self) -> None:
        """
        Visualizes embeddings using the provided visualization manager.
        """
        try:
            results = self.db_manager.fetch_all_embeddings()
            embeddings = [
                np.frombuffer(result[3], dtype=np.float32) for result in results
            ]
            if embeddings:
                matrix = np.stack(embeddings)
                kmeans_labels, _ = self.data_analysis_manager.perform_kmeans_clustering(
                    matrix, n_clusters=5
                )
                self.visualization_manager.visualize_basic_data(matrix, kmeans_labels)
            else:
                messagebox.showinfo(
                    "No Embeddings", "No embeddings found to visualize."
                )
        except Exception as e:
            logging.error(f"Error visualizing embeddings: {e}")
            messagebox.showerror(
                "Visualization Error",
                f"An error occurred during visualization: {str(e)}",
            )
