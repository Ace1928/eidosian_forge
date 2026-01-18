from __future__ import unicode_literals
from prompt_toolkit.enums import IncrementalSearchDirection, SEARCH_BUFFER
from prompt_toolkit.selection import PasteMode
from six.moves import range
import six
from .completion import generate_completions, display_completions_like_readline
from prompt_toolkit.document import Document
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.key_binding.input_processor import KeyPress
from prompt_toolkit.keys import Keys
@register('yank-pop')
def yank_pop(event):
    """
    Rotate the kill ring, and yank the new top. Only works following yank or
    yank-pop.
    """
    buff = event.current_buffer
    doc_before_paste = buff.document_before_paste
    clipboard = event.cli.clipboard
    if doc_before_paste is not None:
        buff.document = doc_before_paste
        clipboard.rotate()
        buff.paste_clipboard_data(clipboard.get_data(), paste_mode=PasteMode.EMACS)