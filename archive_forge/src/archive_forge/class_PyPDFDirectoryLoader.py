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
class PyPDFDirectoryLoader(BaseLoader):
    """Load a directory with `PDF` files using `pypdf` and chunks at character level.

    Loader also stores page numbers in metadata.
    """

    def __init__(self, path: Union[str, Path], glob: str='**/[!.]*.pdf', silent_errors: bool=False, load_hidden: bool=False, recursive: bool=False, extract_images: bool=False):
        self.path = path
        self.glob = glob
        self.load_hidden = load_hidden
        self.recursive = recursive
        self.silent_errors = silent_errors
        self.extract_images = extract_images

    @staticmethod
    def _is_visible(path: Path) -> bool:
        return not any((part.startswith('.') for part in path.parts))

    def load(self) -> List[Document]:
        p = Path(self.path)
        docs = []
        items = p.rglob(self.glob) if self.recursive else p.glob(self.glob)
        for i in items:
            if i.is_file():
                if self._is_visible(i.relative_to(p)) or self.load_hidden:
                    try:
                        loader = PyPDFLoader(str(i), extract_images=self.extract_images)
                        sub_docs = loader.load()
                        for doc in sub_docs:
                            doc.metadata['source'] = str(i)
                        docs.extend(sub_docs)
                    except Exception as e:
                        if self.silent_errors:
                            logger.warning(e)
                        else:
                            raise e
        return docs