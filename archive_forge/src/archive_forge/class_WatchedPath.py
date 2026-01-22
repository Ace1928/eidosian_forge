from __future__ import annotations
import os
import threading
from typing import Callable, Final, cast
from blinker import ANY, Signal
from watchdog import events
from watchdog.observers import Observer
from watchdog.observers.api import ObservedWatch
from streamlit.logger import get_logger
from streamlit.util import repr_
from streamlit.watcher import util
class WatchedPath:
    """Emits notifications when a single path is modified."""

    def __init__(self, md5: str, modification_time: float, *, glob_pattern: str | None=None, allow_nonexistent: bool=False):
        self.md5 = md5
        self.modification_time = modification_time
        self.glob_pattern = glob_pattern
        self.allow_nonexistent = allow_nonexistent
        self.on_changed = Signal()

    def __repr__(self) -> str:
        return repr_(self)