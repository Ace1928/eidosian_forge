import functools
import json
import multiprocessing
import os
import threading
from contextlib import contextmanager
from threading import Thread
from ._colorizer import Colorizer
from ._locks_machinery import create_handler_lock
@contextmanager
def _protected_lock(self):
    """Acquire the lock, but fail fast if its already acquired by the current thread."""
    if getattr(self._lock_acquired, 'acquired', False):
        raise RuntimeError("Could not acquire internal lock because it was already in use (deadlock avoided). This likely happened because the logger was re-used inside a sink, a signal handler or a '__del__' method. This is not permitted because the logger and its handlers are not re-entrant.")
    self._lock_acquired.acquired = True
    try:
        with self._lock:
            yield
    finally:
        self._lock_acquired.acquired = False