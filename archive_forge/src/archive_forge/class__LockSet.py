import errno
import threading
from time import sleep
import weakref
class _LockSet(object):

    def __init__(self, locks):
        self._locks = locks

    def acquire(self, timeout, retry_period):
        acquired_locks = []
        try:
            for lock in self._locks:
                lock.acquire(timeout, retry_period)
                acquired_locks.append(lock)
        except:
            for acquired_lock in reversed(acquired_locks):
                acquired_lock.release()
            raise

    def release(self):
        for lock in reversed(self._locks):
            lock.release()