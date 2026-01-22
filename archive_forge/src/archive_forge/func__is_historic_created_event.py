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
def _is_historic_created_event(self, event):
    in_history = event.inode in self._fs_view
    if self._starting_state:
        try:
            old_inode = self._starting_state.inode(event.path)[0]
            before_start = old_inode == event.inode
        except KeyError:
            before_start = False
    else:
        before_start = False
    return in_history or before_start