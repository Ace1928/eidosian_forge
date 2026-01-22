from __future__ import annotations
import json
import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import guard_import
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
class TableParams:
    """Baidu VectorDB table params.

    See the following documentation for details:
    https://cloud.baidu.com/doc/VDB/s/mlrsob0p6
    """

    def __init__(self, dimension: int, replication: int=3, partition: int=1, index_type: str='HNSW', metric_type: str='L2', params: Optional[Dict]=None):
        self.dimension = dimension
        self.replication = replication
        self.partition = partition
        self.index_type = index_type
        self.metric_type = metric_type
        self.params = params