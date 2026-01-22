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
@handle('escape', '*', filter=insert_mode)
def _insert_all_completions(event: E) -> None:
    """
        `meta-*`: Insert all possible completions of the preceding text.
        """
    buff = event.current_buffer
    complete_event = CompleteEvent(text_inserted=False, completion_requested=True)
    completions = list(buff.completer.get_completions(buff.document, complete_event))
    text_to_insert = ' '.join((c.text for c in completions))
    buff.insert_text(text_to_insert)