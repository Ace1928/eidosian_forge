import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_community.document_loaders.base import BaseLoader
def _load_documents_from_paths(self) -> List[Document]:
    """Load documents from a list of Dropbox file paths."""
    if not self.dropbox_file_paths:
        raise ValueError('file_paths must be set')
    return [doc for doc in (self._load_file_from_path(file_path) for file_path in self.dropbox_file_paths) if doc is not None]