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
class RedisVectorStoreRetriever(VectorStoreRetriever):
    """Retriever for Redis VectorStore."""
    vectorstore: Redis
    'Redis VectorStore.'
    search_type: str = 'similarity'
    "Type of search to perform. Can be either\n    'similarity',\n    'similarity_distance_threshold',\n    'similarity_score_threshold'\n    "
    search_kwargs: Dict[str, Any] = {'k': 4, 'score_threshold': 0.9, 'distance_threshold': None}
    'Default search kwargs.'
    allowed_search_types = ['similarity', 'similarity_distance_threshold', 'similarity_score_threshold', 'mmr']
    'Allowed search types.'

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        if self.search_type == 'similarity':
            docs = self.vectorstore.similarity_search(query, **self.search_kwargs)
        elif self.search_type == 'similarity_distance_threshold':
            if self.search_kwargs['distance_threshold'] is None:
                raise ValueError('distance_threshold must be provided for ' + 'similarity_distance_threshold retriever')
            docs = self.vectorstore.similarity_search(query, **self.search_kwargs)
        elif self.search_type == 'similarity_score_threshold':
            docs_and_similarities = self.vectorstore.similarity_search_with_relevance_scores(query, **self.search_kwargs)
            docs = [doc for doc, _ in docs_and_similarities]
        elif self.search_type == 'mmr':
            docs = self.vectorstore.max_marginal_relevance_search(query, **self.search_kwargs)
        else:
            raise ValueError(f'search_type of {self.search_type} not allowed.')
        return docs

    async def _aget_relevant_documents(self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun) -> List[Document]:
        if self.search_type == 'similarity':
            docs = await self.vectorstore.asimilarity_search(query, **self.search_kwargs)
        elif self.search_type == 'similarity_distance_threshold':
            if self.search_kwargs['distance_threshold'] is None:
                raise ValueError('distance_threshold must be provided for ' + 'similarity_distance_threshold retriever')
            docs = await self.vectorstore.asimilarity_search(query, **self.search_kwargs)
        elif self.search_type == 'similarity_score_threshold':
            docs_and_similarities = await self.vectorstore.asimilarity_search_with_relevance_scores(query, **self.search_kwargs)
            docs = [doc for doc, _ in docs_and_similarities]
        elif self.search_type == 'mmr':
            docs = await self.vectorstore.amax_marginal_relevance_search(query, **self.search_kwargs)
        else:
            raise ValueError(f'search_type of {self.search_type} not allowed.')
        return docs

    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Add documents to vectorstore."""
        return self.vectorstore.add_documents(documents, **kwargs)

    async def aadd_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Add documents to vectorstore."""
        return await self.vectorstore.aadd_documents(documents, **kwargs)