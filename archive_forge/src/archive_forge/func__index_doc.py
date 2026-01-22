from __future__ import annotations
import json
import logging
import os
from dataclasses import dataclass, field
from hashlib import md5
from typing import Any, Iterable, List, Optional, Tuple, Type
import requests
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import Field
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
def _index_doc(self, doc: dict, use_core_api: bool=False) -> str:
    request: dict[str, Any] = {}
    request['customer_id'] = self._vectara_customer_id
    request['corpus_id'] = self._vectara_corpus_id
    request['document'] = doc
    api_endpoint = 'https://api.vectara.io/v1/core/index' if use_core_api else 'https://api.vectara.io/v1/index'
    response = self._session.post(headers=self._get_post_headers(), url=api_endpoint, data=json.dumps(request), timeout=self.vectara_api_timeout, verify=True)
    status_code = response.status_code
    result = response.json()
    status_str = result['status']['code'] if 'status' in result else None
    if status_code == 409 or (status_str and status_str == 'ALREADY_EXISTS'):
        return 'E_ALREADY_EXISTS'
    elif status_str and status_str == 'FORBIDDEN':
        return 'E_NO_PERMISSIONS'
    else:
        return 'E_SUCCEEDED'