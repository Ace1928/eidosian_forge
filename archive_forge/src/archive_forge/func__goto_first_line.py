from __future__ import annotations
import codecs
import string
from enum import Enum
from itertools import accumulate
from typing import Callable, Iterable, Tuple, TypeVar
from prompt_toolkit.application.current import get_app
from prompt_toolkit.buffer import Buffer, indent, reshape_text, unindent
from prompt_toolkit.clipboard import ClipboardData
from prompt_toolkit.document import Document
from prompt_toolkit.filters import (
from prompt_toolkit.filters.app import (
from prompt_toolkit.input.vt100_parser import Vt100Parser
from prompt_toolkit.key_binding.digraphs import DIGRAPHS
from prompt_toolkit.key_binding.key_processor import KeyPress, KeyPressEvent
from prompt_toolkit.key_binding.vi_state import CharacterFind, InputMode
from prompt_toolkit.keys import Keys
from prompt_toolkit.search import SearchDirection
from prompt_toolkit.selection import PasteMode, SelectionState, SelectionType
from ..key_bindings import ConditionalKeyBindings, KeyBindings, KeyBindingsBase
from .named_commands import get_by_name
@text_object('g', 'g')
def _goto_first_line(event: E) -> TextObject:
    """
        Go to the start of the very first line.
        Implements 'gg', 'cgg', 'ygg'
        """
    d = event.current_buffer.document
    if event._arg:
        return TextObject(d.translate_row_col_to_index(event.arg - 1, 0) - d.cursor_position, type=TextObjectType.LINEWISE)
    else:
        return TextObject(d.get_start_of_document_position(), type=TextObjectType.LINEWISE)