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
class PyPDFLoader(BasePDFLoader):
    """Load PDF using pypdf into list of documents.

    Loader chunks by page and stores page numbers in metadata.
    """

    def __init__(self, file_path: str, password: Optional[Union[str, bytes]]=None, headers: Optional[Dict]=None, extract_images: bool=False) -> None:
        """Initialize with a file path."""
        try:
            import pypdf
        except ImportError:
            raise ImportError('pypdf package not found, please install it with `pip install pypdf`')
        super().__init__(file_path, headers=headers)
        self.parser = PyPDFParser(password=password, extract_images=extract_images)

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load given path as pages."""
        if self.web_path:
            blob = Blob.from_data(open(self.file_path, 'rb').read(), path=self.web_path)
        else:
            blob = Blob.from_path(self.file_path)
        yield from self.parser.parse(blob)