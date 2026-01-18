from __future__ import unicode_literals
from .auto_suggest import AutoSuggest
from .clipboard import ClipboardData
from .completion import Completer, Completion, CompleteEvent
from .document import Document
from .enums import IncrementalSearchDirection
from .filters import to_simple_filter
from .history import History, InMemoryHistory
from .search_state import SearchState
from .selection import SelectionType, SelectionState, PasteMode
from .utils import Event
from .cache import FastDictCache
from .validation import ValidationError
from six.moves import range
import os
import re
import six
import subprocess
import tempfile
def transform_current_line(self, transform_callback):
    """
        Apply the given transformation function to the current line.

        :param transform_callback: callable that takes a string and return a new string.
        """
    document = self.document
    a = document.cursor_position + document.get_start_of_line_position()
    b = document.cursor_position + document.get_end_of_line_position()
    self.text = document.text[:a] + transform_callback(document.text[a:b]) + document.text[b:]