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
def _upload_to_gcs(self, data: str, gcs_location: str) -> None:
    """Uploads data to gcs_location.

        Args:
            data: The data that will be stored.
            gcs_location: The location where the data will be stored.
        """
    bucket = self.gcs_client.get_bucket(self.gcs_bucket_name)
    blob = bucket.blob(gcs_location)
    blob.upload_from_string(data)