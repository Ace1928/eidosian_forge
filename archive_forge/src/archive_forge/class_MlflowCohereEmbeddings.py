from __future__ import annotations
from typing import Any, Dict, Iterator, List
from urllib.parse import urlparse
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, PrivateAttr
class MlflowCohereEmbeddings(MlflowEmbeddings):
    """Cohere embedding LLMs in MLflow."""
    query_params: Dict[str, str] = {'input_type': 'search_query'}
    documents_params: Dict[str, str] = {'input_type': 'search_document'}