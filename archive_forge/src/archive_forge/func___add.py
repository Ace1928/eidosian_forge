from __future__ import annotations
import operator
import pickle
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.docstore.base import AddableMixin, Docstore
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.utils import DistanceStrategy
def __add(self, texts: Iterable[str], embeddings: Iterable[List[float]], metadatas: Optional[List[dict]]=None, ids: Optional[List[str]]=None, **kwargs: Any) -> List[str]:
    if not isinstance(self.docstore, AddableMixin):
        raise ValueError(f'If trying to add texts, the underlying docstore should support adding items, which {self.docstore} does not')
    raise NotImplementedError('Updates are not available in ScaNN, yet.')