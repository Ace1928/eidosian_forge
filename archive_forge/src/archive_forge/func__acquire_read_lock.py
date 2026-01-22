import collections
import contextlib
import functools
import threading
from typing import Optional
from fasteners import _utils
def _acquire_read_lock(self, me):
    if me in self._pending_writers:
        raise RuntimeError('Writer %s can not acquire a read lock while waiting for the write lock' % me)
    with self._cond:
        while True:
            if self._writer is None or self._writer == me:
                if me in self._readers:
                    self._readers[me] = self._readers[me] + 1
                    break
                elif self._writer == me or not self.has_pending_writers:
                    self._readers[me] = 1
                    break
            self._cond.wait()