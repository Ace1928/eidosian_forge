import contextlib
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple
from urllib.parse import unquote
from langchain_core.documents import Document
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.web_base import WebBaseLoader
def _get_attachments(self, soup: Any) -> List[str]:
    """Get all attachments from a page.

        Args:
            soup: BeautifulSoup4 soup object.

        Returns:
            List of attachments.
        """
    from bs4 import BeautifulSoup, Tag
    content_list: BeautifulSoup
    content_list = soup.find('ul', {'class': 'contentList'})
    if content_list is None:
        raise ValueError('No content list found.')
    attachments = []
    attachment: Tag
    for attachment in content_list.find_all('ul', {'class': 'attachments'}):
        link: Tag
        for link in attachment.find_all('a'):
            href = link.get('href')
            if href is not None and (not href.startswith('#')):
                attachments.append(href)
    return attachments