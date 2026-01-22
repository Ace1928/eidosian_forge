from __future__ import annotations
import json
import logging
import uuid
from typing import Any, Iterable, List, Optional, Tuple, Type, cast
import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def _query_post(self, query_str: str, cache_name: str, local: bool=False) -> requests.Response:
    api_url = self._default_node + self._cache_url + '/' + cache_name + '?action=search&local=' + str(local)
    data = {'query': query_str}
    data_json = json.dumps(data)
    response = requests.post(api_url, data_json, headers={'Content-Type': 'application/json'}, timeout=REST_TIMEOUT)
    return response