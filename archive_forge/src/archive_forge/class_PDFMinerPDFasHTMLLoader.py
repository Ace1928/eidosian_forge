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
class PDFMinerPDFasHTMLLoader(BasePDFLoader):
    """Load `PDF` files as HTML content using `PDFMiner`."""

    def __init__(self, file_path: str, *, headers: Optional[Dict]=None):
        """Initialize with a file path."""
        try:
            from pdfminer.high_level import extract_text_to_fp
        except ImportError:
            raise ImportError('`pdfminer` package not found, please install it with `pip install pdfminer.six`')
        super().__init__(file_path, headers=headers)

    def lazy_load(self) -> Iterator[Document]:
        """Load file."""
        from pdfminer.high_level import extract_text_to_fp
        from pdfminer.layout import LAParams
        from pdfminer.utils import open_filename
        output_string = StringIO()
        with open_filename(self.file_path, 'rb') as fp:
            extract_text_to_fp(fp, output_string, codec='', laparams=LAParams(), output_type='html')
        metadata = {'source': self.file_path if self.web_path is None else self.web_path}
        yield Document(page_content=output_string.getvalue(), metadata=metadata)