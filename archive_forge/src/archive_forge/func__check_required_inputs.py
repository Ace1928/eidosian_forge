from __future__ import annotations
import base64
import logging
import uuid
from copy import deepcopy
from typing import (
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _check_required_inputs(self, collection_name: str) -> None:
    if not self._client.is_connected():
        raise ValueError('VDMS client must be connected to a VDMS server.' + 'Please use VDMS_Client to establish a connection')
    if self.distance_strategy not in AVAILABLE_DISTANCE_METRICS:
        raise ValueError("distance_strategy must be either 'L2' or 'IP'")
    if self.similarity_search_engine not in AVAILABLE_ENGINES:
        raise ValueError("engine must be either 'TileDBDense', 'TileDBSparse', " + "'FaissFlat', 'FaissIVFFlat', or 'Flinng'")
    if self.embedding is None:
        raise ValueError('Must provide embedding function')
    self.embedding_dimension = len(self._embed_query('This is a sample sentence.'))
    current_props = self.__get_properties(collection_name)
    if hasattr(self, 'collection_properties'):
        self.collection_properties.extend(current_props)
    else:
        self.collection_properties: List[str] = current_props