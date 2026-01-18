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
@register('yank-last-arg')
def yank_last_arg(event):
    """
    Like `yank_nth_arg`, but if no argument has been given, yank the last word
    of each line.
    """
    n = event.arg if event.arg_present else None
    event.current_buffer.yank_last_arg(n)