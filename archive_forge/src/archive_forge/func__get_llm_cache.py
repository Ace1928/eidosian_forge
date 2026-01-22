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
def _get_llm_cache(self, llm_string: str) -> AzureCosmosDBVectorSearch:
    index_name = self._index_name(llm_string)
    namespace = self.database_name + '.' + self.collection_name
    if index_name in self._cache_dict:
        return self._cache_dict[index_name]
    if self.cosmosdb_client:
        collection = self.cosmosdb_client[self.database_name][self.collection_name]
        self._cache_dict[index_name] = AzureCosmosDBVectorSearch(collection=collection, embedding=self.embedding, index_name=index_name)
    else:
        self._cache_dict[index_name] = AzureCosmosDBVectorSearch.from_connection_string(connection_string=self.cosmosdb_connection_string, namespace=namespace, embedding=self.embedding, index_name=index_name, application_name=self.application_name)
    vectorstore = self._cache_dict[index_name]
    if not vectorstore.index_exists():
        vectorstore.create_index(self.num_lists, self.dimensions, self.similarity, self.kind, self.m, self.ef_construction)
    return vectorstore