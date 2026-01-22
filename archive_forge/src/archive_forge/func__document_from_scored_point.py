from __future__ import annotations
import functools
import uuid
import warnings
from itertools import islice
from operator import itemgetter
from typing import (
import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_core.vectorstores import VectorStore
from langchain_community.docstore.document import Document
from langchain_community.vectorstores.utils import maximal_marginal_relevance
@classmethod
def _document_from_scored_point(cls, scored_point: Any, collection_name: str, content_payload_key: str, metadata_payload_key: str) -> Document:
    metadata = scored_point.payload.get(metadata_payload_key) or {}
    metadata['_id'] = scored_point.id
    metadata['_collection_name'] = collection_name
    return Document(page_content=scored_point.payload.get(content_payload_key), metadata=metadata)