from typing import Iterator
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_community.document_loaders.blob_loaders import Blob
Parse a Microsoft Word document into the Document iterator.

        Args:
            blob: The blob to parse.

        Returns: An iterator of Documents.

        