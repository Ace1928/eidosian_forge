import logging
from typing import Any, Iterator, List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
def _generate_docs_page(self, result: Any) -> Iterator[Document]:
    for p in result.pages:
        content = ' '.join([line.content for line in p.lines])
        d = Document(page_content=content, metadata={'page': p.page_number})
        yield d