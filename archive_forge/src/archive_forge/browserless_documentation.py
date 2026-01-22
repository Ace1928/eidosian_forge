from typing import Iterator, List, Union
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
Lazy load Documents from URLs.