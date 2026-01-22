from __future__ import annotations
import io
import json
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
Initialize Athena document loader.

        Args:
            query: The query to run in Athena.
            database: Athena database
            s3_output_uri: Athena output path
            metadata_columns: Optional. Columns written to Document `metadata`.
        