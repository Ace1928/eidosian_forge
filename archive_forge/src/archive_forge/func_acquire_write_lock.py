from contextlib import contextmanager
import dbm
import os
import threading
from ..api import BytesBackend
from ..api import NO_VALUE
from ... import util
def acquire_write_lock(self, wait):
    return self._acquire(wait, os.O_WRONLY, self._module.LOCK_EX)