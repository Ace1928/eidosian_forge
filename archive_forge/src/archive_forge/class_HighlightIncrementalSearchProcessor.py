from a buffer before the BufferControl will render it to the screen.
from __future__ import annotations
import re
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Callable, Hashable, cast
from prompt_toolkit.application.current import get_app
from prompt_toolkit.cache import SimpleCache
from prompt_toolkit.document import Document
from prompt_toolkit.filters import FilterOrBool, to_filter, vi_insert_multiple_mode
from prompt_toolkit.formatted_text import (
from prompt_toolkit.formatted_text.utils import fragment_list_len, fragment_list_to_text
from prompt_toolkit.search import SearchDirection
from prompt_toolkit.utils import to_int, to_str
from .utils import explode_text_fragments
class HighlightIncrementalSearchProcessor(HighlightSearchProcessor):
    """
    Highlight the search terms that are used for highlighting the incremental
    search. The style class 'incsearch' will be applied to the content.

    Important: this requires the `preview_search=True` flag to be set for the
    `BufferControl`. Otherwise, the cursor position won't be set to the search
    match while searching, and nothing happens.
    """
    _classname = 'incsearch'
    _classname_current = 'incsearch.current'

    def _get_search_text(self, buffer_control: BufferControl) -> str:
        """
        The text we are searching for.
        """
        search_buffer = buffer_control.search_buffer
        if search_buffer is not None and search_buffer.text:
            return search_buffer.text
        return ''