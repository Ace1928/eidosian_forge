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
def _find_property_entity(collection_name: str, unique_entity: Optional[bool]=False, deletion: Optional[bool]=False) -> Dict[str, Dict[str, Any]]:
    querytype = 'FindEntity'
    entity: Dict[str, Any] = {}
    entity['class'] = 'properties'
    if unique_entity:
        entity['unique'] = unique_entity
    results: Dict[str, Any] = {}
    results['blob'] = True
    results['count'] = ''
    results['list'] = ['content']
    entity['results'] = results
    constraints: Dict[str, Any] = {}
    if deletion:
        constraints['_deletion'] = ['==', 1]
    constraints['name'] = ['==', collection_name]
    entity['constraints'] = constraints
    query: Dict[str, Any] = {}
    query[querytype] = entity
    return query