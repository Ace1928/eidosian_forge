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
@handle('s-left', filter=shift_selection_mode)
@handle('s-right', filter=shift_selection_mode)
@handle('s-up', filter=shift_selection_mode)
@handle('s-down', filter=shift_selection_mode)
@handle('s-home', filter=shift_selection_mode)
@handle('s-end', filter=shift_selection_mode)
@handle('c-s-left', filter=shift_selection_mode)
@handle('c-s-right', filter=shift_selection_mode)
@handle('c-s-home', filter=shift_selection_mode)
@handle('c-s-end', filter=shift_selection_mode)
def _extend_selection(event: E) -> None:
    """
        Extend the selection
        """
    unshift_move(event)
    buff = event.current_buffer
    if buff.selection_state is not None:
        if buff.cursor_position == buff.selection_state.original_cursor_position:
            buff.exit_selection()