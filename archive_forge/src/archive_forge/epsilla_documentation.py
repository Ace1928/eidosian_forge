from __future__ import annotations
import logging
import uuid
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Type
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
Create an Epsilla vectorstore from a list of documents.

        Args:
            texts (List[str]): List of text data to be inserted.
            embeddings (Embeddings): Embedding function.
            client (pyepsilla.vectordb.Client): Epsilla client to connect to.
            metadatas (Optional[List[dict]]): Metadata for each text.
                    Defaults to None.
            db_path (Optional[str]): The path where the database will be persisted.
                    Defaults to "/tmp/langchain-epsilla".
            db_name (Optional[str]): Give a name to the loaded database.
                    Defaults to "langchain_store".
            collection_name (Optional[str]): Which collection to use.
                    Defaults to "langchain_collection".
                    If provided, default collection name will be set as well.
            drop_old (Optional[bool]): Whether to drop the previous collection
                    and create a new one. Defaults to False.

        Returns:
            Epsilla: Epsilla vector store.
        