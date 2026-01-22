import threading
import fasteners
from fasteners import test
class RWLocked(object):

    def __init__(self):
        self._lock = fasteners.ReaderWriterLock()

    @fasteners.read_locked
    def i_am_read_locked(self, cb):
        cb(self._lock.owner)

    @fasteners.write_locked
    def i_am_write_locked(self, cb):
        cb(self._lock.owner)

    def i_am_not_locked(self, cb):
        cb(self._lock.owner)