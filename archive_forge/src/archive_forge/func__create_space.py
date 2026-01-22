from __future__ import annotations
import os
import time
import uuid
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Type
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def _create_space(self, dim: int=1024) -> int:
    """
        Create VectorStore space
        Args:
            dim:dimension of vector
        Return:
            code,0 failed for ,1 for success
        """
    space_config = {'name': self.using_table_name, 'partition_num': 1, 'replica_num': 1, 'engine': {'name': 'gamma', 'index_size': 1, 'retrieval_type': 'FLAT', 'retrieval_param': {'metric_type': 'L2'}}, 'properties': {'text': {'type': 'string'}, 'metadata': {'type': 'string'}, 'text_embedding': {'type': 'vector', 'index': True, 'dimension': dim, 'store_type': 'MemoryOnly'}}}
    response_code = self.vearch.create_space(self.using_db_name, space_config)
    return response_code