import sys
import ctypes
import threading
import importlib.resources as _impres
from llvmlite.binding.common import _decode_string, _is_shutting_down
from llvmlite.utils import get_library_name
class _LLVMLock:
    """A Lock to guarantee thread-safety for the LLVM C-API.

    This class implements __enter__ and __exit__ for acquiring and releasing
    the lock as a context manager.

    Also, callbacks can be attached so that every time the lock is acquired
    and released the corresponding callbacks will be invoked.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._cblist = []

    def register(self, acq_fn, rel_fn):
        """Register callbacks that are invoked immediately after the lock is
        acquired (``acq_fn()``) and immediately before the lock is released
        (``rel_fn()``).
        """
        self._cblist.append((acq_fn, rel_fn))

    def unregister(self, acq_fn, rel_fn):
        """Remove the registered callbacks.
        """
        self._cblist.remove((acq_fn, rel_fn))

    def __enter__(self):
        self._lock.acquire()
        for acq_fn, rel_fn in self._cblist:
            acq_fn()

    def __exit__(self, *exc_details):
        for acq_fn, rel_fn in self._cblist:
            rel_fn()
        self._lock.release()