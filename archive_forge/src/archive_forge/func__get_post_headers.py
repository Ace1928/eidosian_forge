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
def _get_post_headers(self) -> dict:
    """Returns headers that should be attached to each post request."""
    return {'x-api-key': self._vectara_api_key, 'customer-id': self._vectara_customer_id, 'Content-Type': 'application/json', 'X-Source': self._source}