import errno
import threading
from time import sleep
import weakref
class _ThreadLock(object):

    def __init__(self, path):
        self._path = path
        self._lock = threading.Lock()

    def acquire(self, timeout=None, retry_period=None):
        if timeout is None:
            self._lock.acquire()
        else:
            _acquire_non_blocking(acquire=lambda: self._lock.acquire(False), timeout=timeout, retry_period=retry_period, path=self._path)

    def release(self):
        self._lock.release()