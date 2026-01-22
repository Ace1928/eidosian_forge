from __future__ import annotations
import sys
import threading
import time
import weakref
from typing import Any, Callable, Optional
from pymongo.lock import _create_lock
def _register_executor(executor: PeriodicExecutor) -> None:
    ref = weakref.ref(executor, _on_executor_deleted)
    _EXECUTORS.add(ref)