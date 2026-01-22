from contextlib import contextmanager
import dbm
import os
import threading
from ..api import BytesBackend
from ..api import NO_VALUE
from ... import util
@contextmanager
def _use_rw_lock(self, write):
    if self._rw_lock is None:
        yield
    elif write:
        with self._rw_lock.write():
            yield
    else:
        with self._rw_lock.read():
            yield