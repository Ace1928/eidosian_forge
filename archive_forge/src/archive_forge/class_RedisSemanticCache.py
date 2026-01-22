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
class RedisSemanticCache(BaseCache):
    """Cache that uses Redis as a vector-store backend."""
    DEFAULT_SCHEMA = {'content_key': 'prompt', 'text': [{'name': 'prompt'}], 'extra': [{'name': 'return_val'}, {'name': 'llm_string'}]}

    def __init__(self, redis_url: str, embedding: Embeddings, score_threshold: float=0.2):
        """Initialize by passing in the `init` GPTCache func

        Args:
            redis_url (str): URL to connect to Redis.
            embedding (Embedding): Embedding provider for semantic encoding and search.
            score_threshold (float, 0.2):

        Example:

        .. code-block:: python

            from langchain_community.globals import set_llm_cache

            from langchain_community.cache import RedisSemanticCache
            from langchain_community.embeddings import OpenAIEmbeddings

            set_llm_cache(RedisSemanticCache(
                redis_url="redis://localhost:6379",
                embedding=OpenAIEmbeddings()
            ))

        """
        self._cache_dict: Dict[str, RedisVectorstore] = {}
        self.redis_url = redis_url
        self.embedding = embedding
        self.score_threshold = score_threshold

    def _index_name(self, llm_string: str) -> str:
        hashed_index = _hash(llm_string)
        return f'cache:{hashed_index}'

    def _get_llm_cache(self, llm_string: str) -> RedisVectorstore:
        index_name = self._index_name(llm_string)
        if index_name in self._cache_dict:
            return self._cache_dict[index_name]
        try:
            self._cache_dict[index_name] = RedisVectorstore.from_existing_index(embedding=self.embedding, index_name=index_name, redis_url=self.redis_url, schema=cast(Dict, self.DEFAULT_SCHEMA))
        except ValueError:
            redis = RedisVectorstore(embedding=self.embedding, index_name=index_name, redis_url=self.redis_url, index_schema=cast(Dict, self.DEFAULT_SCHEMA))
            _embedding = self.embedding.embed_query(text='test')
            redis._create_index_if_not_exist(dim=len(_embedding))
            self._cache_dict[index_name] = redis
        return self._cache_dict[index_name]

    def clear(self, **kwargs: Any) -> None:
        """Clear semantic cache for a given llm_string."""
        index_name = self._index_name(kwargs['llm_string'])
        if index_name in self._cache_dict:
            self._cache_dict[index_name].drop_index(index_name=index_name, delete_documents=True, redis_url=self.redis_url)
            del self._cache_dict[index_name]

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        llm_cache = self._get_llm_cache(llm_string)
        generations: List = []
        results = llm_cache.similarity_search(query=prompt, k=1, distance_threshold=self.score_threshold)
        if results:
            for document in results:
                try:
                    generations.extend(loads(document.metadata['return_val']))
                except Exception:
                    logger.warning('Retrieving a cache value that could not be deserialized properly. This is likely due to the cache being in an older format. Please recreate your cache to avoid this error.')
                    generations.extend(_load_generations_from_json(document.metadata['return_val']))
        return generations if generations else None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""
        for gen in return_val:
            if not isinstance(gen, Generation):
                raise ValueError(f'RedisSemanticCache only supports caching of normal LLM generations, got {type(gen)}')
        llm_cache = self._get_llm_cache(llm_string)
        metadata = {'llm_string': llm_string, 'prompt': prompt, 'return_val': dumps([g for g in return_val])}
        llm_cache.add_texts(texts=[prompt], metadatas=[metadata])