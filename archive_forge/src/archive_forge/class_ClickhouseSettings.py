from __future__ import annotations
import json
import logging
from hashlib import sha1
from threading import Thread
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseSettings
from langchain_core.vectorstores import VectorStore
class ClickhouseSettings(BaseSettings):
    """`ClickHouse` client configuration.

    Attribute:
        host (str) : An URL to connect to MyScale backend.
                             Defaults to 'localhost'.
        port (int) : URL port to connect with HTTP. Defaults to 8443.
        username (str) : Username to login. Defaults to None.
        password (str) : Password to login. Defaults to None.
        index_type (str): index type string.
        index_param (list): index build parameter.
        index_query_params(dict): index query parameters.
        database (str) : Database name to find the table. Defaults to 'default'.
        table (str) : Table name to operate on.
                      Defaults to 'vector_table'.
        metric (str) : Metric to compute distance,
                       supported are ('angular', 'euclidean', 'manhattan', 'hamming',
                       'dot'). Defaults to 'angular'.
                       https://github.com/spotify/annoy/blob/main/src/annoymodule.cc#L149-L169

        column_map (Dict) : Column type map to project column name onto langchain
                            semantics. Must have keys: `text`, `id`, `vector`,
                            must be same size to number of columns. For example:
                            .. code-block:: python

                                {
                                    'id': 'text_id',
                                    'uuid': 'global_unique_id'
                                    'embedding': 'text_embedding',
                                    'document': 'text_plain',
                                    'metadata': 'metadata_dictionary_in_json',
                                }

                            Defaults to identity map.
    """
    host: str = 'localhost'
    port: int = 8123
    username: Optional[str] = None
    password: Optional[str] = None
    index_type: str = 'annoy'
    index_param: Optional[Union[List, Dict]] = ["'L2Distance'", 100]
    index_query_params: Dict[str, str] = {}
    column_map: Dict[str, str] = {'id': 'id', 'uuid': 'uuid', 'document': 'document', 'embedding': 'embedding', 'metadata': 'metadata'}
    database: str = 'default'
    table: str = 'langchain'
    metric: str = 'angular'

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    class Config:
        env_file = '.env'
        env_prefix = 'clickhouse_'
        env_file_encoding = 'utf-8'