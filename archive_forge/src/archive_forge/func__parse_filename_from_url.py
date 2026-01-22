import contextlib
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple
from urllib.parse import unquote
from langchain_core.documents import Document
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.web_base import WebBaseLoader
def _parse_filename_from_url(self, url: str) -> str:
    """Parse the filename from an url.

        Args:
            url: Url to parse the filename from.

        Returns:
            The filename.

        Raises:
            ValueError: If the filename could not be parsed.
        """
    filename_matches = re.search('filename%2A%3DUTF-8%27%27(.+)', url)
    if filename_matches:
        filename = filename_matches.group(1)
    else:
        raise ValueError(f'Could not parse filename from {url}')
    if '.pdf' not in filename:
        raise ValueError(f'Incorrect file type: {filename}')
    filename = filename.split('.pdf')[0] + '.pdf'
    filename = unquote(filename)
    filename = filename.replace('%20', ' ')
    return filename