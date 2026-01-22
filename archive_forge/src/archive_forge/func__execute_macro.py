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
@handle('@', Keys.Any, filter=vi_navigation_mode, record_in_macro=False)
def _execute_macro(event: E) -> None:
    """
        Execute macro.

        Notice that we pass `record_in_macro=False`. This ensures that the `@x`
        keys don't appear in the recording itself. This function inserts the
        body of the called macro back into the KeyProcessor, so these keys will
        be added later on to the macro of their handlers have
        `record_in_macro=True`.
        """
    c = event.key_sequence[1].data
    try:
        macro = event.app.vi_state.named_registers[c]
    except KeyError:
        return
    keys: list[KeyPress] = []
    parser = Vt100Parser(keys.append)
    parser.feed(macro.text)
    parser.flush()
    for _ in range(event.arg):
        event.app.key_processor.feed_multiple(keys, first=True)