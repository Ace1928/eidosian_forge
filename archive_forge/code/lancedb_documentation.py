from __future__ import annotations
import uuid
from typing import Any, Iterable, List, Optional
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
Return documents most similar to the query

        Args:
            query: String to query the vectorstore with.
            k: Number of documents to return.

        Returns:
            List of documents most similar to the query.
        