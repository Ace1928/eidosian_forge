from __future__ import annotations
from prompt_toolkit.application.current import get_app
from prompt_toolkit.buffer import Buffer, indent, unindent
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.filters import (
from prompt_toolkit.key_binding.key_bindings import Binding
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.selection import SelectionType
from ..key_bindings import ConditionalKeyBindings, KeyBindings, KeyBindingsBase
from .named_commands import get_by_name
@handle('s-left', filter=~has_selection)
@handle('s-right', filter=~has_selection)
@handle('s-up', filter=~has_selection)
@handle('s-down', filter=~has_selection)
@handle('s-home', filter=~has_selection)
@handle('s-end', filter=~has_selection)
@handle('c-s-left', filter=~has_selection)
@handle('c-s-right', filter=~has_selection)
@handle('c-s-home', filter=~has_selection)
@handle('c-s-end', filter=~has_selection)
def _start_selection(event: E) -> None:
    """
        Start selection with shift + movement.
        """
    buff = event.current_buffer
    if buff.text:
        buff.start_selection(selection_type=SelectionType.CHARACTERS)
        if buff.selection_state is not None:
            buff.selection_state.enter_shift_mode()
        original_position = buff.cursor_position
        unshift_move(event)
        if buff.cursor_position == original_position:
            buff.exit_selection()