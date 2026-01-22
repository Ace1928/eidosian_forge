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
def apply_completion(self, completion):
    """
        Insert a given completion.
        """
    assert isinstance(completion, Completion)
    if self.complete_state:
        self.go_to_completion(None)
    self.complete_state = None
    self.delete_before_cursor(-completion.start_position)
    self.insert_text(completion.text)