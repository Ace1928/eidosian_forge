from __future__ import annotations
import os
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Type
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
Return docs according ids.

        Args:
            ids: The ids of the embedding vectors.
        Returns:
            Documents which satisfy the input conditions.
        