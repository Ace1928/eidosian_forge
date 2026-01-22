from __future__ import annotations
import json
import logging
import uuid
from typing import Any, Iterable, List, Optional, Type
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
def _uuid_key() -> str:
    return uuid.uuid4().hex