from __future__ import annotations
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Optional, Tuple, Type
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VST, VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _infer_embedding_dimension(self) -> int:
    """Infer the embedding dimension from the embedding function."""
    assert self.embeddings is not None, 'embedding model is required.'
    return len(self.embeddings.embed_query('test'))