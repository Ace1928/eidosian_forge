from __future__ import annotations
import logging
import os
import uuid
from typing import (
import numpy as np
import yaml
from langchain_core._api import deprecated
from langchain_core.callbacks import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_community.utilities.redis import (
from langchain_community.vectorstores.redis.constants import (
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def _prepare_range_query(self, k: int, filter: Optional[RedisFilterExpression]=None, return_fields: Optional[List[str]]=None) -> 'Query':
    try:
        from redis.commands.search.query import Query
    except ImportError as e:
        raise ImportError('Could not import redis python package. Please install it with `pip install redis`.') from e
    return_fields = return_fields or []
    vector_key = self._schema.content_vector_key
    base_query = f'@{vector_key}:[VECTOR_RANGE $distance_threshold $vector]'
    if filter:
        base_query = str(filter) + ' ' + base_query
    query_string = base_query + '=>{$yield_distance_as: distance}'
    return Query(query_string).return_fields(*return_fields).sort_by('distance').paging(0, k).dialect(2)