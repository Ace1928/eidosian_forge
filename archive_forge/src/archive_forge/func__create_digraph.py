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
@handle(Keys.Any, filter=vi_digraph_mode & digraph_symbol_1_given)
def _create_digraph(event: E) -> None:
    """
        Insert digraph.
        """
    try:
        code: tuple[str, str] = (event.app.vi_state.digraph_symbol1 or '', event.data)
        if code not in DIGRAPHS:
            code = code[::-1]
        symbol = DIGRAPHS[code]
    except KeyError:
        event.app.output.bell()
    else:
        overwrite = event.app.vi_state.input_mode == InputMode.REPLACE
        event.current_buffer.insert_text(chr(symbol), overwrite=overwrite)
        event.app.vi_state.waiting_for_digraph = False
    finally:
        event.app.vi_state.waiting_for_digraph = False
        event.app.vi_state.digraph_symbol1 = None