from __future__ import annotations
import os
import threading
import weakref
def _release_locks() -> None:
    for lock in _forkable_locks:
        if lock.locked():
            lock.release()