import tkinter as tk
from tkinter import ttk
import ast
import json
import nltk
from nltk.corpus import wordnet
import black  # Python code formatter
import requests
from functools import lru_cache

# Ensure nltk resources are downloaded
nltk.download("wordnet")
nltk.download("omw-1.4")

# Local modules developed for specific tasks
from semantic_analyzer import SemanticAnalyzer
from lexical_analyzer import LexicalAnalyzer
from code_analyzer import CodeAnalyzer
from code_formatter import CodeFormatter
from synonym_replacer import SynonymReplacer
from naming_convention import NamingConventionEnforcer
from embeddings_database import EmbeddingsDatabase


class CodeStandardizer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Python Code Standardizer")
        self.geometry("1200x600")

        self.original_code_text = tk.Text(self, height=30, width=60)
        self.original_code_text.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.standardized_code_text = tk.Text(self, height=30, width=60)
        self.standardized_code_text.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        self.run_button = ttk.Button(
            self, text="Standardize Code", command=self.standardize_code
        )
        self.run_button.pack(side=tk.BOTTOM, pady=10)

    def standardize_code(self):
        original_code = self.original_code_text.get("1.0", tk.END)
        self.standardized_code_text.delete("1.0", tk.END)
        standardized_code = self.apply_standardization(original_code)
        self.standardized_code_text.insert("1.0", standardized_code)

    def apply_standardization(self, code: str) -> str:
        # Semantic and syntactic analysis using local NLP models
        semantic_analyzer = SemanticAnalyzer()
        lexical_analyzer = LexicalAnalyzer()
        semantic_results = semantic_analyzer.analyze(code)
        lexical_results = lexical_analyzer.analyze(code)

        # Analyze code using local tools
        code_analyzer = CodeAnalyzer()
        analysis_results = code_analyzer.analyze_code(code)
        suggestions = json.dumps(analysis_results, indent=4)

        # Applying Python code formatting using local tools
        code_formatter = CodeFormatter()
        formatted_code = code_formatter.format_code(code)
        formatted_code = black.format_str(formatted_code, mode=black.FileMode())

        # Synonym replacement for better naming conventions
        synonym_replacer = SynonymReplacer()
        formatted_code = synonym_replacer.replace_synonyms(formatted_code)

        # Enforce naming conventions
        naming_convention_enforcer = NamingConventionEnforcer()
        formatted_code = naming_convention_enforcer.enforce(formatted_code)

        # Save code and analysis to embeddings vector database for knowledge graph construction
        embeddings_database = EmbeddingsDatabase()
        embeddings_database.save_code_embeddings(
            formatted_code, semantic_results, lexical_results
        )

        # Combine automated formatting results with LLM suggestions.
        combined_results = f"{formatted_code}\n\nSemantic Analysis:\n{semantic_results}\n\nLexical Analysis:\n{lexical_results}\n\nLLM Suggestions:\n{suggestions}"
        return combined_results

    def run(self):
        self.mainloop()


if __name__ == "__main__":
    app = CodeStandardizer()
    app.run()
