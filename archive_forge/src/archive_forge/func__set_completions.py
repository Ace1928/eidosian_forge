from __future__ import annotations
import asyncio
import logging
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from collections import deque
from enum import Enum
from functools import wraps
from typing import Any, Callable, Coroutine, Iterable, TypeVar, cast
from .application.current import get_app
from .application.run_in_terminal import run_in_terminal
from .auto_suggest import AutoSuggest, Suggestion
from .cache import FastDictCache
from .clipboard import ClipboardData
from .completion import (
from .document import Document
from .eventloop import aclosing
from .filters import FilterOrBool, to_filter
from .history import History, InMemoryHistory
from .search import SearchDirection, SearchState
from .selection import PasteMode, SelectionState, SelectionType
from .utils import Event, to_str
from .validation import ValidationError, Validator
def _set_completions(self, completions: list[Completion]) -> CompletionState:
    """
        Start completions. (Generate list of completions and initialize.)

        By default, no completion will be selected.
        """
    self.complete_state = CompletionState(original_document=self.document, completions=completions)
    self.on_completions_changed.fire()
    return self.complete_state