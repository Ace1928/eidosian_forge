from __future__ import annotations
import asyncio
import logging
import re
from typing import (
import requests
from langchain_core.documents import Document
from langchain_core.utils.html import extract_sub_links
from langchain_community.document_loaders.base import BaseLoader
def _metadata_extractor(raw_html: str, url: str) -> dict:
    """Extract metadata from raw html using BeautifulSoup."""
    metadata = {'source': url}
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.warning('The bs4 package is required for default metadata extraction. Please install it with `pip install bs4`.')
        return metadata
    soup = BeautifulSoup(raw_html, 'html.parser')
    if (title := soup.find('title')):
        metadata['title'] = title.get_text()
    if (description := soup.find('meta', attrs={'name': 'description'})):
        metadata['description'] = description.get('content', None)
    if (html := soup.find('html')):
        metadata['language'] = html.get('lang', None)
    return metadata