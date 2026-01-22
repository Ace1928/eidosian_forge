from __future__ import annotations
import logging
import os
import threading
import time
import unicodedata
import _watchdog_fsevents as _fsevents  # type: ignore[import-not-found]
from watchdog.events import (
from watchdog.observers.api import DEFAULT_EMITTER_TIMEOUT, DEFAULT_OBSERVER_TIMEOUT, BaseObserver, EventEmitter
from watchdog.utils.dirsnapshot import DirectorySnapshot
def _is_recursive_event(self, event):
    src_path = event.src_path if event.is_directory else os.path.dirname(event.src_path)
    if src_path == self._absolute_watch_path:
        return False
    if isinstance(event, (FileMovedEvent, DirMovedEvent)):
        dest_path = os.path.dirname(event.dest_path)
        if dest_path == self._absolute_watch_path:
            return False
    return True