import datetime
import json
from pathlib import Path
from typing import Iterator, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
class FacebookChatLoader(BaseLoader):
    """Load `Facebook Chat` messages directory dump."""

    def __init__(self, path: Union[str, Path]):
        """Initialize with a path."""
        self.file_path = path

    def lazy_load(self) -> Iterator[Document]:
        p = Path(self.file_path)
        with open(p, encoding='utf8') as f:
            d = json.load(f)
        text = ''.join((concatenate_rows(message) for message in d['messages'] if message.get('content') and isinstance(message['content'], str)))
        metadata = {'source': str(p)}
        yield Document(page_content=text, metadata=metadata)