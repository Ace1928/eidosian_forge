from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import string
from prompt_toolkit.filters import IsReadOnly
from prompt_toolkit.key_binding.bindings.vi import create_operator_decorator
from prompt_toolkit.key_binding.vi_state import InputMode
from prompt_toolkit.keys import Keys
import six
@operator(*handler_keys, filter=~IsReadOnly())
def ChangeOperator(event, text_object):
    """A change operator."""
    clipboard_data = None
    buf = event.current_buffer
    if text_object:
        if text_object.start < text_object.end:
            while text_object.end > text_object.start and buf.text[buf.cursor_position + text_object.end - 1].isspace():
                text_object.end -= 1
        else:
            while text_object.start > text_object.end and buf.text[buf.cursor_position + text_object.start - 1].isspace():
                text_object.start -= 1
        new_document, clipboard_data = text_object.cut(buf)
        buf.document = new_document
    if clipboard_data and clipboard_data.text:
        if with_register:
            reg_name = event.key_sequence[1].data
            if reg_name in vi_register_names:
                event.cli.vi_state.named_registers[reg_name] = clipboard_data
        else:
            event.cli.clipboard.set_data(clipboard_data)
    event.cli.vi_state.input_mode = InputMode.INSERT