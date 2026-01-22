from __future__ import annotations
import os
from typing import (
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def _format_toots(self, toots: List[Dict[str, Any]], user_info: dict) -> Iterable[Document]:
    """Format toots into documents.

        Adding user info, and selected toot fields into the metadata.
        """
    for toot in toots:
        metadata = {'created_at': toot['created_at'], 'user_info': user_info, 'is_reply': toot['in_reply_to_id'] is not None}
        yield Document(page_content=toot['content'], metadata=metadata)