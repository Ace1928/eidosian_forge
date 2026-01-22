from __future__ import annotations
import time
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Callable, Hashable, Iterable, NamedTuple
from prompt_toolkit.application.current import get_app
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.cache import SimpleCache
from prompt_toolkit.data_structures import Point
from prompt_toolkit.document import Document
from prompt_toolkit.filters import FilterOrBool, to_filter
from prompt_toolkit.formatted_text import (
from prompt_toolkit.formatted_text.utils import (
from prompt_toolkit.lexers import Lexer, SimpleLexer
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from prompt_toolkit.search import SearchState
from prompt_toolkit.selection import SelectionType
from prompt_toolkit.utils import get_cwidth
from .processors import (
def _get_formatted_text_for_line_func(self, document: Document) -> Callable[[int], StyleAndTextTuples]:
    """
        Create a function that returns the fragments for a given line.
        """

    def get_formatted_text_for_line() -> Callable[[int], StyleAndTextTuples]:
        return self.lexer.lex_document(document)
    key = (document.text, self.lexer.invalidation_hash())
    return self._fragment_cache.get(key, get_formatted_text_for_line)