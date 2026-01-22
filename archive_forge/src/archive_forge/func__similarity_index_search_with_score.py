import json
import math
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Type
from uuid import uuid4
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import guard_import
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _similarity_index_search_with_score(self, query_embedding: List[float], *, k: int=DEFAULT_K, **kwargs: Any) -> List[Tuple[int, float]]:
    """Search k embeddings similar to the query embedding. Returns a list of
        (index, distance) tuples."""
    if not self._neighbors_fitted:
        raise SKLearnVectorStoreException('No data was added to SKLearnVectorStore.')
    neigh_dists, neigh_idxs = self._neighbors.kneighbors([query_embedding], n_neighbors=k)
    return list(zip(neigh_idxs[0], neigh_dists[0]))