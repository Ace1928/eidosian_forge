from __future__ import annotations
import sys
import threading
import time
import weakref
from typing import Any, Callable, Optional
from pymongo.lock import _create_lock
def update_interval(self, new_interval: int) -> None:
    self._interval = new_interval