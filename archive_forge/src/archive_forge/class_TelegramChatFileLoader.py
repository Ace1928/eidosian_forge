from __future__ import annotations
import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
class TelegramChatFileLoader(BaseLoader):
    """Load from `Telegram chat` dump."""

    def __init__(self, path: Union[str, Path]):
        """Initialize with a path."""
        self.file_path = path

    def load(self) -> List[Document]:
        """Load documents."""
        p = Path(self.file_path)
        with open(p, encoding='utf8') as f:
            d = json.load(f)
        text = ''.join((concatenate_rows(message) for message in d['messages'] if message['type'] == 'message' and isinstance(message['text'], str)))
        metadata = {'source': str(p)}
        return [Document(page_content=text, metadata=metadata)]