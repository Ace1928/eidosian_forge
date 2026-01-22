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
def apply_search(self, search_state, include_current_position=True, count=1):
    """
        Apply search. If something is found, set `working_index` and
        `cursor_position`.
        """
    search_result = self._search(search_state, include_current_position=include_current_position, count=count)
    if search_result is not None:
        working_index, cursor_position = search_result
        self.working_index = working_index
        self.cursor_position = cursor_position