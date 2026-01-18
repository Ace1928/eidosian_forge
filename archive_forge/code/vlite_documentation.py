from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
Construct VLite wrapper from a list of documents.

        This is a user-friendly interface that:
        1. Embeds documents.
        2. Adds the documents to the vectorstore.

        This is intended to be a quick way to get started.

        Example:
        .. code-block:: python

            from langchain import VLite
            from langchain.embeddings import OpenAIEmbeddings

            embeddings = OpenAIEmbeddings()
            vlite = VLite.from_documents(documents, embeddings)
        