from __future__ import annotations
import uuid
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Type
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def _check_bucket_exists(self) -> bool:
    """Check if the bucket exists in the linked Couchbase cluster"""
    bucket_manager = self._cluster.buckets()
    try:
        bucket_manager.get_bucket(self._bucket_name)
        return True
    except Exception:
        return False