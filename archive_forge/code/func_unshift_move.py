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
def unshift_move(event: E) -> None:
    """
        Used for the shift selection mode. When called with
        a shift + movement key press event, moves the cursor
        as if shift is not pressed.
        """
    key = event.key_sequence[0].key
    if key == Keys.ShiftUp:
        event.current_buffer.auto_up(count=event.arg)
        return
    if key == Keys.ShiftDown:
        event.current_buffer.auto_down(count=event.arg)
        return
    key_to_command: dict[Keys | str, str] = {Keys.ShiftLeft: 'backward-char', Keys.ShiftRight: 'forward-char', Keys.ShiftHome: 'beginning-of-line', Keys.ShiftEnd: 'end-of-line', Keys.ControlShiftLeft: 'backward-word', Keys.ControlShiftRight: 'forward-word', Keys.ControlShiftHome: 'beginning-of-buffer', Keys.ControlShiftEnd: 'end-of-buffer'}
    try:
        binding = get_by_name(key_to_command[key])
    except KeyError:
        pass
    else:
        if isinstance(binding, Binding):
            binding.call(event)