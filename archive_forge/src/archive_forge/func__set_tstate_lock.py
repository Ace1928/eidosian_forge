import os as _os
import sys as _sys
import _thread
import functools
from time import monotonic as _time
from _weakrefset import WeakSet
from itertools import islice as _islice, count as _count
from _thread import stack_size
def _set_tstate_lock(self):
    """
        Set a lock object which will be released by the interpreter when
        the underlying thread state (see pystate.h) gets deleted.
        """
    self._tstate_lock = _set_sentinel()
    self._tstate_lock.acquire()
    if not self.daemon:
        with _shutdown_locks_lock:
            _maintain_shutdown_locks()
            _shutdown_locks.add(self._tstate_lock)