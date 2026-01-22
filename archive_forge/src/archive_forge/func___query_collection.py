from __future__ import annotations
import asyncio
import enum
import json
import logging
import struct
import uuid
from collections import OrderedDict
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseSettings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import maximal_marginal_relevance
def __query_collection(self, embedding: List[float], k: int=4, filter: Optional[Dict[str, str]]=None) -> Dict:
    """Query the collection."""
    json_filter = json.dumps(filter) if filter is not None else None
    where_clause = f" where '{json_filter}' = JSON(metadata) " if json_filter is not None else ''
    embedding_str = '[' + ','.join([str(x) for x in embedding]) + ']'
    dist_strategy = self.distance_strategy
    query_string = f"\n                SELECT text, metadata, {dist_strategy}(embedding, '{embedding_str}') \n                as distance, embedding\n                FROM {self.table_name}\n                {where_clause}\n                ORDER BY distance asc NULLS LAST\n                LIMIT {k}\n        "
    self.logger.debug(query_string)
    resp = self._db.execute_sql_and_decode(query_string)
    self.logger.debug(resp)
    return resp