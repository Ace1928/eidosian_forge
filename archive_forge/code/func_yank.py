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
@register('yank')
def yank(event):
    """
    Paste before cursor.
    """
    event.current_buffer.paste_clipboard_data(event.cli.clipboard.get_data(), count=event.arg, paste_mode=PasteMode.EMACS)