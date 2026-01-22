from __future__ import annotations
import sys
import threading
import time
import weakref
from typing import Any, Callable, Optional
from pymongo.lock import _create_lock
def __should_stop(self) -> bool:
    with self._lock:
        if self._stopped:
            self._thread_will_exit = True
            return True
        return False