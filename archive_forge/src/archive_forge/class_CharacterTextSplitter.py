from __future__ import annotations
import re
from typing import Any, List, Optional
from langchain_text_splitters.base import Language, TextSplitter
class CharacterTextSplitter(TextSplitter):
    """Splitting text that looks at characters."""

    def __init__(self, separator: str='\n\n', is_separator_regex: bool=False, **kwargs: Any) -> None:
        """Create a new TextSplitter."""
        super().__init__(**kwargs)
        self._separator = separator
        self._is_separator_regex = is_separator_regex

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        separator = self._separator if self._is_separator_regex else re.escape(self._separator)
        splits = _split_text_with_regex(text, separator, self._keep_separator)
        _separator = '' if self._keep_separator else self._separator
        return self._merge_splits(splits, _separator)