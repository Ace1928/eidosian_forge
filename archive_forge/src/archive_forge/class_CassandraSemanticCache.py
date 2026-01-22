from __future__ import annotations
import hashlib
import inspect
import json
import logging
import uuid
import warnings
from abc import ABC
from datetime import timedelta
from enum import Enum
from functools import lru_cache, wraps
from typing import (
from sqlalchemy import Column, Integer, String, create_engine, delete, select
from sqlalchemy.engine import Row
from sqlalchemy.engine.base import Engine
from sqlalchemy.orm import Session
from langchain_community.vectorstores.azure_cosmos_db import (
from langchain_core._api.deprecation import deprecated
from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM, aget_prompts, get_prompts
from langchain_core.load.dump import dumps
from langchain_core.load.load import loads
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.utils import get_from_env
from langchain_community.utilities.astradb import (
from langchain_community.vectorstores import AzureCosmosDBVectorSearch
from langchain_community.vectorstores.redis import Redis as RedisVectorstore
class CassandraSemanticCache(BaseCache):
    """
    Cache that uses Cassandra as a vector-store backend for semantic
    (i.e. similarity-based) lookup.

    It uses a single (vector) Cassandra table and stores, in principle,
    cached values from several LLMs, so the LLM's llm_string is part
    of the rows' primary keys.

    The similarity is based on one of several distance metrics (default: "dot").
    If choosing another metric, the default threshold is to be re-tuned accordingly.
    """

    def __init__(self, session: Optional[CassandraSession], keyspace: Optional[str], embedding: Embeddings, table_name: str=CASSANDRA_SEMANTIC_CACHE_DEFAULT_TABLE_NAME, distance_metric: str=CASSANDRA_SEMANTIC_CACHE_DEFAULT_DISTANCE_METRIC, score_threshold: float=CASSANDRA_SEMANTIC_CACHE_DEFAULT_SCORE_THRESHOLD, ttl_seconds: Optional[int]=CASSANDRA_SEMANTIC_CACHE_DEFAULT_TTL_SECONDS, skip_provisioning: bool=False):
        """
        Initialize the cache with all relevant parameters.
        Args:
            session (cassandra.cluster.Session): an open Cassandra session
            keyspace (str): the keyspace to use for storing the cache
            embedding (Embedding): Embedding provider for semantic
                encoding and search.
            table_name (str): name of the Cassandra (vector) table
                to use as cache
            distance_metric (str, 'dot'): which measure to adopt for
                similarity searches
            score_threshold (optional float): numeric value to use as
                cutoff for the similarity searches
            ttl_seconds (optional int): time-to-live for cache entries
                (default: None, i.e. forever)
        The default score threshold is tuned to the default metric.
        Tune it carefully yourself if switching to another distance metric.
        """
        try:
            from cassio.table import MetadataVectorCassandraTable
        except (ImportError, ModuleNotFoundError):
            raise ValueError('Could not import cassio python package. Please install it with `pip install cassio`.')
        self.session = session
        self.keyspace = keyspace
        self.embedding = embedding
        self.table_name = table_name
        self.distance_metric = distance_metric
        self.score_threshold = score_threshold
        self.ttl_seconds = ttl_seconds

        @lru_cache(maxsize=CASSANDRA_SEMANTIC_CACHE_EMBEDDING_CACHE_SIZE)
        def _cache_embedding(text: str) -> List[float]:
            return self.embedding.embed_query(text=text)
        self._get_embedding = _cache_embedding
        self.embedding_dimension = self._get_embedding_dimension()
        self.table = MetadataVectorCassandraTable(session=self.session, keyspace=self.keyspace, table=self.table_name, primary_key_type=['TEXT'], vector_dimension=self.embedding_dimension, ttl_seconds=self.ttl_seconds, metadata_indexing=('allow', {'_llm_string_hash'}), skip_provisioning=skip_provisioning)

    def _get_embedding_dimension(self) -> int:
        return len(self._get_embedding(text='This is a sample sentence.'))

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""
        embedding_vector = self._get_embedding(text=prompt)
        llm_string_hash = _hash(llm_string)
        body = _dumps_generations(return_val)
        metadata = {'_prompt': prompt, '_llm_string_hash': llm_string_hash}
        row_id = f'{_hash(prompt)}-{llm_string_hash}'
        self.table.put(body_blob=body, vector=embedding_vector, row_id=row_id, metadata=metadata)

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        hit_with_id = self.lookup_with_id(prompt, llm_string)
        if hit_with_id is not None:
            return hit_with_id[1]
        else:
            return None

    def lookup_with_id(self, prompt: str, llm_string: str) -> Optional[Tuple[str, RETURN_VAL_TYPE]]:
        """
        Look up based on prompt and llm_string.
        If there are hits, return (document_id, cached_entry)
        """
        prompt_embedding: List[float] = self._get_embedding(text=prompt)
        hits = list(self.table.metric_ann_search(vector=prompt_embedding, metadata={'_llm_string_hash': _hash(llm_string)}, n=1, metric=self.distance_metric, metric_threshold=self.score_threshold))
        if hits:
            hit = hits[0]
            generations = _loads_generations(hit['body_blob'])
            if generations is not None:
                return (hit['row_id'], generations)
            else:
                return None
        else:
            return None

    def lookup_with_id_through_llm(self, prompt: str, llm: LLM, stop: Optional[List[str]]=None) -> Optional[Tuple[str, RETURN_VAL_TYPE]]:
        llm_string = get_prompts({**llm.dict(), **{'stop': stop}}, [])[1]
        return self.lookup_with_id(prompt, llm_string=llm_string)

    def delete_by_document_id(self, document_id: str) -> None:
        """
        Given this is a "similarity search" cache, an invalidation pattern
        that makes sense is first a lookup to get an ID, and then deleting
        with that ID. This is for the second step.
        """
        self.table.delete(row_id=document_id)

    def clear(self, **kwargs: Any) -> None:
        """Clear the *whole* semantic cache."""
        self.table.clear()