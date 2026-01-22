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
def _update_neighbors(self) -> None:
    if len(self._embeddings) == 0:
        raise SKLearnVectorStoreException('No data was added to SKLearnVectorStore.')
    self._embeddings_np = self._np.asarray(self._embeddings)
    self._neighbors.fit(self._embeddings_np)
    self._neighbors_fitted = True