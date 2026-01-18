from __future__ import annotations
import base64
import logging
import uuid
from copy import deepcopy
from typing import (
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def update_documents(self, collection_name: str, ids: List[str], documents: List[Document]) -> None:
    """Update a document in the collection.

        Args:
            ids (List[str]): List of ids of the document to update.
            documents (List[Document]): List of documents to update.
        """
    text = [document.page_content for document in documents]
    metadata = [_validate_vdms_properties(document.metadata) for document in documents]
    embeddings = self._embed_documents(text)
    self.__update(collection_name, ids, metadatas=metadata, embeddings=embeddings, documents=text)