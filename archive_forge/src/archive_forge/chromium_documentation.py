import asyncio
import logging
from typing import Iterator, List
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader

        Lazily load text content from the provided URLs.

        This method yields Documents one at a time as they're scraped,
        instead of waiting to scrape all URLs before returning.

        Yields:
            Document: The scraped content encapsulated within a Document object.

        