from __future__ import annotations
from typing import Any, Dict, List, Tuple, TypedDict
from langchain_core.documents import Document
from langchain_text_splitters.base import Language
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
class MarkdownTextSplitter(RecursiveCharacterTextSplitter):
    """Attempts to split the text along Markdown-formatted headings."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a MarkdownTextSplitter."""
        separators = self.get_separators_for_language(Language.MARKDOWN)
        super().__init__(separators=separators, **kwargs)