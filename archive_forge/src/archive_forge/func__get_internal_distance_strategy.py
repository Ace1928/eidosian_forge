from typing import Any, Iterable, List, Optional, Tuple
from uuid import uuid4
import numpy as np
import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_env
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import DistanceStrategy
def _get_internal_distance_strategy(self) -> str:
    """Return the internal distance strategy."""
    if self.distance_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE:
        return 'euclidean'
    elif self.distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
        raise ValueError('Max inner product is not supported by SemaDB')
    elif self.distance_strategy == DistanceStrategy.DOT_PRODUCT:
        return 'dot'
    elif self.distance_strategy == DistanceStrategy.JACCARD:
        raise ValueError('Max inner product is not supported by SemaDB')
    elif self.distance_strategy == DistanceStrategy.COSINE:
        return 'cosine'
    else:
        raise ValueError(f'Unknown distance strategy {self.distance_strategy}')