from __future__ import annotations
import sys
import threading
import time
import weakref
from typing import Any, Callable, Optional
from pymongo.lock import _create_lock
def _shutdown_executors() -> None:
    if _EXECUTORS is None:
        return
    executors = list(_EXECUTORS)
    for ref in executors:
        executor = ref()
        if executor:
            executor.close()
    for ref in executors:
        executor = ref()
        if executor:
            executor.join(1)
    executor = None