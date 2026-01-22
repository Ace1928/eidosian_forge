import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterator, List
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator
def _dict2document(self, doc: dict) -> Document:
    summary = doc.pop('Summary')
    return Document(page_content=summary, metadata=doc)