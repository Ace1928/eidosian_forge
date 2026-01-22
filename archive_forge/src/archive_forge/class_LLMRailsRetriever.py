from __future__ import annotations
import json
import logging
import os
import uuid
from typing import Any, Iterable, List, Optional, Tuple
import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import Field
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
class LLMRailsRetriever(VectorStoreRetriever):
    """Retriever for LLMRails."""
    vectorstore: LLMRails
    search_kwargs: dict = Field(default_factory=lambda: {'k': 5})
    'Search params.\n        k: Number of Documents to return. Defaults to 5.\n        alpha: parameter for hybrid search .\n    '

    def add_texts(self, texts: List[str]) -> None:
        """Add text to the datastore.

        Args:
            texts (List[str]): The text
        """
        self.vectorstore.add_texts(texts)