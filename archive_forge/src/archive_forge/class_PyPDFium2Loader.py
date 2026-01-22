import json
import logging
import os
import re
import tempfile
import time
from abc import ABC
from io import StringIO
from pathlib import Path
from typing import (
from urllib.parse import urlparse
import requests
from langchain_core.documents import Document
from langchain_core.utils import get_from_dict_or_env
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.blob_loaders import Blob
from langchain_community.document_loaders.parsers.pdf import (
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
class PyPDFium2Loader(BasePDFLoader):
    """Load `PDF` using `pypdfium2` and chunks at character level."""

    def __init__(self, file_path: str, *, headers: Optional[Dict]=None, extract_images: bool=False):
        """Initialize with a file path."""
        super().__init__(file_path, headers=headers)
        self.parser = PyPDFium2Parser(extract_images=extract_images)

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load given path as pages."""
        if self.web_path:
            blob = Blob.from_data(open(self.file_path, 'rb').read(), path=self.web_path)
        else:
            blob = Blob.from_path(self.file_path)
        yield from self.parser.parse(blob)