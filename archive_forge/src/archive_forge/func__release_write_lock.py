import collections
import contextlib
import functools
import threading
from typing import Optional
from fasteners import _utils
def _release_write_lock(self, me, raise_on_not_owned=True):
    with self._cond:
        self._writer = None
        self._writer_entries = 0
        self._cond.notify_all()