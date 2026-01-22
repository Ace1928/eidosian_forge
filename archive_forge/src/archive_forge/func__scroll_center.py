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
@handle('z', 'z', filter=vi_navigation_mode | vi_selection_mode)
def _scroll_center(event: E) -> None:
    """
        Center Window vertically around cursor.
        """
    w = event.app.layout.current_window
    b = event.current_buffer
    if w and w.render_info:
        info = w.render_info
        scroll_height = info.window_height // 2
        y = max(0, b.document.cursor_position_row - 1)
        height = 0
        while y > 0:
            line_height = info.get_height_for_line(y)
            if height + line_height < scroll_height:
                height += line_height
                y -= 1
            else:
                break
        w.vertical_scroll = y