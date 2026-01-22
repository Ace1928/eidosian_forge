import asyncio
import contextlib
import contextvars
import threading
from typing import Any, Dict, Union
@contextlib.contextmanager
def _lock_storage(self):
    if self._thread_critical:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            yield self._storage
        else:
            if not hasattr(self._storage, 'cvar'):
                self._storage.cvar = _CVar()
            yield self._storage.cvar
    else:
        with self._thread_lock:
            yield self._storage