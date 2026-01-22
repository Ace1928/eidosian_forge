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
@classmethod
def _create_credentials_from_file(cls, json_credentials_path: Optional[str]) -> Optional[Credentials]:
    """Creates credentials for GCP.

        Args:
             json_credentials_path: The path on the file system where the
             credentials are stored.

         Returns:
             An optional of Credentials or None, in which case the default
             will be used.
        """
    from google.oauth2 import service_account
    credentials = None
    if json_credentials_path is not None:
        credentials = service_account.Credentials.from_service_account_file(json_credentials_path)
    return credentials