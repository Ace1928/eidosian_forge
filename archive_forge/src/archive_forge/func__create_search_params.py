from __future__ import annotations
import logging
from typing import Any, Iterable, List, Optional, Tuple, Union
from uuid import uuid4
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _create_search_params(self) -> None:
    """Generate search params based on the current index type"""
    from pymilvus import Collection
    if isinstance(self.col, Collection) and self.search_params is None:
        index = self._get_index()
        if index is not None:
            index_type: str = index['index_param']['index_type']
            metric_type: str = index['index_param']['metric_type']
            self.search_params = self.default_search_params[index_type]
            self.search_params['metric_type'] = metric_type