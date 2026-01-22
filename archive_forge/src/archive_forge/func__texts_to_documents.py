from __future__ import annotations
import uuid
import warnings
from itertools import repeat
from typing import (
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
@staticmethod
def _texts_to_documents(texts: Iterable[str], metadatas: Optional[Iterable[Dict[Any, Any]]]=None) -> List[Document]:
    """Return list of Documents from list of texts and metadatas."""
    if metadatas is None:
        metadatas = repeat({})
    docs = [Document(page_content=text, metadata=metadata) for text, metadata in zip(texts, metadatas)]
    return docs