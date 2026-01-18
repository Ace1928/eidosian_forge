from __future__ import annotations
import hashlib
import json
import uuid
from itertools import islice
from typing import (
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import root_validator
from langchain_core.vectorstores import VectorStore
from langchain.indexes.base import NAMESPACE_UUID, RecordManager
def to_document(self) -> Document:
    """Return a Document object."""
    return Document(page_content=self.page_content, metadata=self.metadata)