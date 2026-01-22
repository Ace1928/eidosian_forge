from __future__ import annotations
import uuid
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _get_async_opensearch_client(opensearch_url: str, **kwargs: Any) -> Any:
    """Get AsyncOpenSearch client from the opensearch_url, otherwise raise error."""
    try:
        async_opensearch = _import_async_opensearch()
        client = async_opensearch(opensearch_url, **kwargs)
    except ValueError as e:
        raise ImportError(f'AsyncOpenSearch client string provided is not in proper format. Got error: {e} ')
    return client