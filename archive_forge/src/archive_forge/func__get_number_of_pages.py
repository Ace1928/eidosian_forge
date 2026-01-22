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
@staticmethod
def _get_number_of_pages(blob: Blob) -> int:
    try:
        import pypdf
        from PIL import Image, ImageSequence
    except ImportError:
        raise ModuleNotFoundError('Could not import pypdf or Pilloe python package. Please install it with `pip install pypdf Pillow`.')
    if blob.mimetype == 'application/pdf':
        with blob.as_bytes_io() as input_pdf_file:
            pdf_reader = pypdf.PdfReader(input_pdf_file)
            return len(pdf_reader.pages)
    elif blob.mimetype == 'image/tiff':
        num_pages = 0
        img = Image.open(blob.as_bytes())
        for _, _ in enumerate(ImageSequence.Iterator(img)):
            num_pages += 1
        return num_pages
    elif blob.mimetype in ['image/png', 'image/jpeg']:
        return 1
    else:
        raise ValueError(f'unsupported mime type: {blob.mimetype}')