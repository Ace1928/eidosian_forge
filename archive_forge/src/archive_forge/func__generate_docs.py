from __future__ import annotations
import warnings
from typing import (
from urllib.parse import urlparse
import numpy as np
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
def _generate_docs(self, blob: Blob, result: Any) -> Iterator[Document]:
    for p in result.pages:
        content = ' '.join([line.content for line in p.lines])
        d = Document(page_content=content, metadata={'source': blob.source, 'page': p.page_number})
        yield d