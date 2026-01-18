from typing import Iterator, List, Optional
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
Load model information lazily, including README content.