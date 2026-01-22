from __future__ import annotations
import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple, Type
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_community.utilities.vertexai import get_client_info
def _validate_google_libraries_installation(self) -> None:
    """Validates that Google libraries that are needed are installed."""
    try:
        from google.cloud import aiplatform, storage
        from google.oauth2 import service_account
    except ImportError:
        raise ImportError('You must run `pip install --upgrade google-cloud-aiplatform google-cloud-storage`to use the MatchingEngine Vectorstore.')