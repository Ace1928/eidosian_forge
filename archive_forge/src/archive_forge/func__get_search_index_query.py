from __future__ import annotations
import enum
import logging
import os
from hashlib import md5
from typing import (
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import DistanceStrategy
def _get_search_index_query(search_type: SearchType, index_type: IndexType=DEFAULT_INDEX_TYPE) -> str:
    if index_type == IndexType.NODE:
        type_to_query_map = {SearchType.VECTOR: 'CALL db.index.vector.queryNodes($index, $k, $embedding) YIELD node, score ', SearchType.HYBRID: 'CALL { CALL db.index.vector.queryNodes($index, $k, $embedding) YIELD node, score WITH collect({node:node, score:score}) AS nodes, max(score) AS max UNWIND nodes AS n RETURN n.node AS node, (n.score / max) AS score UNION CALL db.index.fulltext.queryNodes($keyword_index, $query, {limit: $k}) YIELD node, score WITH collect({node:node, score:score}) AS nodes, max(score) AS max UNWIND nodes AS n RETURN n.node AS node, (n.score / max) AS score } WITH node, max(score) AS score ORDER BY score DESC LIMIT $k '}
        return type_to_query_map[search_type]
    else:
        return 'CALL db.index.vector.queryRelationships($index, $k, $embedding) YIELD relationship, score '