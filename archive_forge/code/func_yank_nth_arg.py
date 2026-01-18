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
@register('yank-nth-arg')
def yank_nth_arg(event):
    """
    Insert the first argument of the previous command. With an argument, insert
    the nth word from the previous command (start counting at 0).
    """
    n = event.arg if event.arg_present else None
    event.current_buffer.yank_nth_arg(n)