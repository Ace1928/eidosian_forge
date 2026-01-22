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
@handle('escape', 'right')
def _start_next_word(event: E) -> None:
    """
        Cursor to start of next word.
        """
    buffer = event.current_buffer
    buffer.cursor_position += buffer.document.find_next_word_beginning(count=event.arg) or buffer.document.get_end_of_document_position()