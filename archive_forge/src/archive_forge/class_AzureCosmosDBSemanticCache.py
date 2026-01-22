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
class AzureCosmosDBSemanticCache(BaseCache):
    """Cache that uses Cosmos DB Mongo vCore vector-store backend"""
    DEFAULT_DATABASE_NAME = 'CosmosMongoVCoreCacheDB'
    DEFAULT_COLLECTION_NAME = 'CosmosMongoVCoreCacheColl'

    def __init__(self, cosmosdb_connection_string: str, database_name: str, collection_name: str, embedding: Embeddings, *, cosmosdb_client: Optional[Any]=None, num_lists: int=100, similarity: CosmosDBSimilarityType=CosmosDBSimilarityType.COS, kind: CosmosDBVectorSearchType=CosmosDBVectorSearchType.VECTOR_IVF, dimensions: int=1536, m: int=16, ef_construction: int=64, ef_search: int=40, score_threshold: Optional[float]=None, application_name: str='LANGCHAIN_CACHING_PYTHON'):
        """
        Args:
            cosmosdb_connection_string: Cosmos DB Mongo vCore connection string
            cosmosdb_client: Cosmos DB Mongo vCore client
            embedding (Embedding): Embedding provider for semantic encoding and search.
            database_name: Database name for the CosmosDBMongoVCoreSemanticCache
            collection_name: Collection name for the CosmosDBMongoVCoreSemanticCache
            num_lists: This integer is the number of clusters that the
                inverted file (IVF) index uses to group the vector data.
                We recommend that numLists is set to documentCount/1000
                for up to 1 million documents and to sqrt(documentCount)
                for more than 1 million documents.
                Using a numLists value of 1 is akin to performing
                brute-force search, which has limited performance
            dimensions: Number of dimensions for vector similarity.
                The maximum number of supported dimensions is 2000
            similarity: Similarity metric to use with the IVF index.

                Possible options are:
                    - CosmosDBSimilarityType.COS (cosine distance),
                    - CosmosDBSimilarityType.L2 (Euclidean distance), and
                    - CosmosDBSimilarityType.IP (inner product).
            kind: Type of vector index to create.
                Possible options are:
                    - vector-ivf
                    - vector-hnsw: available as a preview feature only,
                                   to enable visit https://learn.microsoft.com/en-us/azure/azure-resource-manager/management/preview-features
            m: The max number of connections per layer (16 by default, minimum
               value is 2, maximum value is 100). Higher m is suitable for datasets
               with high dimensionality and/or high accuracy requirements.
            ef_construction: the size of the dynamic candidate list for constructing
                            the graph (64 by default, minimum value is 4, maximum
                            value is 1000). Higher ef_construction will result in
                            better index quality and higher accuracy, but it will
                            also increase the time required to build the index.
                            ef_construction has to be at least 2 * m
            ef_search: The size of the dynamic candidate list for search
                       (40 by default). A higher value provides better
                       recall at the cost of speed.
            score_threshold: Maximum score used to filter the vector search documents.
            application_name: Application name for the client for tracking and logging
        """
        self._validate_enum_value(similarity, CosmosDBSimilarityType)
        self._validate_enum_value(kind, CosmosDBVectorSearchType)
        if not cosmosdb_connection_string:
            raise ValueError(' CosmosDB connection string can be empty.')
        self.cosmosdb_connection_string = cosmosdb_connection_string
        self.cosmosdb_client = cosmosdb_client
        self.embedding = embedding
        self.database_name = database_name or self.DEFAULT_DATABASE_NAME
        self.collection_name = collection_name or self.DEFAULT_COLLECTION_NAME
        self.num_lists = num_lists
        self.dimensions = dimensions
        self.similarity = similarity
        self.kind = kind
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.score_threshold = score_threshold
        self._cache_dict: Dict[str, AzureCosmosDBVectorSearch] = {}
        self.application_name = application_name

    def _index_name(self, llm_string: str) -> str:
        hashed_index = _hash(llm_string)
        return f'cache:{hashed_index}'

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

    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        llm_cache = self._get_llm_cache(llm_string)
        generations: List = []
        results = llm_cache.similarity_search(query=prompt, k=1, kind=self.kind, ef_search=self.ef_search, score_threshold=self.score_threshold)
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
                raise ValueError(f'CosmosDBMongoVCoreSemanticCache only supports caching of normal LLM generations, got {type(gen)}')
        llm_cache = self._get_llm_cache(llm_string)
        metadata = {'llm_string': llm_string, 'prompt': prompt, 'return_val': dumps([g for g in return_val])}
        llm_cache.add_texts(texts=[prompt], metadatas=[metadata])

    def clear(self, **kwargs: Any) -> None:
        """Clear semantic cache for a given llm_string."""
        index_name = self._index_name(kwargs['llm_string'])
        if index_name in self._cache_dict:
            self._cache_dict[index_name].get_collection().delete_many({})

    @staticmethod
    def _validate_enum_value(value: Any, enum_type: Type[Enum]) -> None:
        if not isinstance(value, enum_type):
            raise ValueError(f'Invalid enum value: {value}. Expected {enum_type}.')