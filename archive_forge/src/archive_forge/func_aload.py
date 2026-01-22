import asyncio
import logging
import warnings
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union
import aiohttp
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def aload(self) -> List[Document]:
    """Load text from the urls in web_path async into Documents."""
    results = self.scrape_all(self.web_paths)
    docs = []
    for path, soup in zip(self.web_paths, results):
        text = soup.get_text(**self.bs_get_text_kwargs)
        metadata = _build_metadata(soup, path)
        docs.append(Document(page_content=text, metadata=metadata))
    return docs