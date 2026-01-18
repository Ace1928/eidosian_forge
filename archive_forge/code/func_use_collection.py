from __future__ import annotations
import logging
import uuid
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Type
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def use_collection(self, collection_name: str) -> None:
    """
        Set default collection to use.

        Args:
            collection_name (str): The name of the collection.
        """
    self._collection_name = collection_name