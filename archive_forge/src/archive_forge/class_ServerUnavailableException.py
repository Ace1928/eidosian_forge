import logging
from typing import Dict, Iterator, List, Union
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
class ServerUnavailableException(Exception):
    """Exception raised when the Grobid server is unavailable."""
    pass