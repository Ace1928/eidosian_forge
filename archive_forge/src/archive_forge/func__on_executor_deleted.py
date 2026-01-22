from __future__ import annotations
import sys
import threading
import time
import weakref
from typing import Any, Callable, Optional
from pymongo.lock import _create_lock
def _on_executor_deleted(ref: weakref.ReferenceType[PeriodicExecutor]) -> None:
    _EXECUTORS.remove(ref)