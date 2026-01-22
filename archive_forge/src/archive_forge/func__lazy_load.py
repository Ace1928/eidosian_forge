import hashlib
import logging
from base64 import b64decode
from pathlib import Path
from time import strptime
from typing import Any, Dict, Iterator, List, Optional, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def _lazy_load(self) -> Iterator[Document]:
    for note in self._parse_note_xml(self.file_path):
        if note.get('content') is not None:
            yield Document(page_content=note['content'], metadata={**{key: value for key, value in note.items() if key not in ['content', 'content-raw', 'resource']}, **{'source': self.file_path}})