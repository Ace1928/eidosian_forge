import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
@property
def tidb_vector_client(self) -> Any:
    """Return the TiDB Vector Client."""
    return self._tidb