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
@handle('backspace', filter=vi_insert_multiple_mode)
def _delete_before_multiple_cursors(event: E) -> None:
    """
        Backspace, using multiple cursors.
        """
    buff = event.current_buffer
    original_text = buff.text
    deleted_something = False
    text = []
    p = 0
    for p2 in buff.multiple_cursor_positions:
        if p2 > 0 and original_text[p2 - 1] != '\n':
            text.append(original_text[p:p2 - 1])
            deleted_something = True
        else:
            text.append(original_text[p:p2])
        p = p2
    text.append(original_text[p:])
    if deleted_something:
        lengths = [len(part) for part in text[:-1]]
        new_cursor_positions = list(accumulate(lengths))
        buff.text = ''.join(text)
        buff.multiple_cursor_positions = new_cursor_positions
        buff.cursor_position -= 1
    else:
        event.app.output.bell()