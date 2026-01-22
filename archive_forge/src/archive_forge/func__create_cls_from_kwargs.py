import logging
import uuid
from abc import ABC, abstractmethod
from typing import (
import numpy as np
from langchain_core._api import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores.utils import (
@staticmethod
def _create_cls_from_kwargs(embedding: Optional[Embeddings]=None, **kwargs: Any) -> 'ElasticsearchStore':
    index_name = kwargs.get('index_name')
    if index_name is None:
        raise ValueError('Please provide an index_name.')
    es_connection = kwargs.get('es_connection')
    es_cloud_id = kwargs.get('es_cloud_id')
    es_url = kwargs.get('es_url')
    es_user = kwargs.get('es_user')
    es_password = kwargs.get('es_password')
    es_api_key = kwargs.get('es_api_key')
    vector_query_field = kwargs.get('vector_query_field')
    query_field = kwargs.get('query_field')
    distance_strategy = kwargs.get('distance_strategy')
    strategy = kwargs.get('strategy', ElasticsearchStore.ApproxRetrievalStrategy())
    optional_args = {}
    if vector_query_field is not None:
        optional_args['vector_query_field'] = vector_query_field
    if query_field is not None:
        optional_args['query_field'] = query_field
    return ElasticsearchStore(index_name=index_name, embedding=embedding, es_url=es_url, es_connection=es_connection, es_cloud_id=es_cloud_id, es_user=es_user, es_password=es_password, es_api_key=es_api_key, strategy=strategy, distance_strategy=distance_strategy, **optional_args)