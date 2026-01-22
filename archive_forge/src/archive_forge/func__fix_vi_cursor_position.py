from __future__ import unicode_literals
from prompt_toolkit.buffer import EditReadOnlyBuffer
from prompt_toolkit.filters.cli import ViNavigationMode
from prompt_toolkit.keys import Keys, Key
from prompt_toolkit.utils import Event
from .registry import BaseRegistry
from collections import deque
from six.moves import range
import weakref
import six
def _fix_vi_cursor_position(self, event):
    """
        After every command, make sure that if we are in Vi navigation mode, we
        never put the cursor after the last character of a line. (Unless it's
        an empty line.)
        """
    cli = self._cli_ref()
    if cli:
        buff = cli.current_buffer
        preferred_column = buff.preferred_column
        if ViNavigationMode()(event.cli) and buff.document.is_cursor_at_the_end_of_line and (len(buff.document.current_line) > 0):
            buff.cursor_position -= 1
            buff.preferred_column = preferred_column