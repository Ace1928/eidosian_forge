from __future__ import annotations
import time
from itertools import repeat
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def _extractMetadata(self, record: dict) -> dict:
    """Extract metadata from a record. Filters out known columns."""
    metadata = {}
    for key, val in record.items():
        if key not in ['id', 'content', 'embedding', 'xata']:
            metadata[key] = val
    return metadata