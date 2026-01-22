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
def _is_s3_presigned_url(url: str) -> bool:
    """Check if the url is a presigned S3 url."""
    try:
        result = urlparse(url)
        return bool(re.search('\\.s3\\.amazonaws\\.com$', result.netloc))
    except ValueError:
        return False