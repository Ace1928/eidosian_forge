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
@handle('c-y', filter=shift_selection_mode)
def _yank(event: E) -> None:
    """
        In shift selection mode, yanking (pasting) replace the selection.
        """
    buff = event.current_buffer
    if buff.selection_state:
        buff.cut_selection()
    get_by_name('yank').call(event)