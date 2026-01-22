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
def _op_require_direct_access_index(self, op_name: str) -> None:
    """
        Raise ValueError if the operation is not supported for direct-access index."""
    if not self._is_direct_access_index():
        raise ValueError(f'`{op_name}` is only supported for direct-access index.')