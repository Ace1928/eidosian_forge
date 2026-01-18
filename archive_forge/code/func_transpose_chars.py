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
@register('transpose-chars')
def transpose_chars(event):
    """
    Emulate Emacs transpose-char behavior: at the beginning of the buffer,
    do nothing.  At the end of a line or buffer, swap the characters before
    the cursor.  Otherwise, move the cursor right, and then swap the
    characters before the cursor.
    """
    b = event.current_buffer
    p = b.cursor_position
    if p == 0:
        return
    elif p == len(b.text) or b.text[p] == '\n':
        b.swap_characters_before_cursor()
    else:
        b.cursor_position += b.document.get_cursor_right_position()
        b.swap_characters_before_cursor()