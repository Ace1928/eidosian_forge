from __future__ import annotations
from typing import Any, Dict, Iterator, List
from urllib.parse import urlparse
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, PrivateAttr
@property
def _mlflow_extras(self) -> str:
    return '[genai]'