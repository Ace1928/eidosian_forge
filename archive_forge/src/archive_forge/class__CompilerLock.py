import threading
import functools
import numba.core.event as ev
class _CompilerLock(object):

    def __init__(self):
        self._lock = threading.RLock()

    def acquire(self):
        ev.start_event('numba:compiler_lock')
        self._lock.acquire()

    def release(self):
        self._lock.release()
        ev.end_event('numba:compiler_lock')

    def __enter__(self):
        self.acquire()

    def __exit__(self, exc_val, exc_type, traceback):
        self.release()

    def is_locked(self):
        is_owned = getattr(self._lock, '_is_owned')
        if not callable(is_owned):
            is_owned = self._is_owned
        return is_owned()

    def __call__(self, func):

        @functools.wraps(func)
        def _acquire_compile_lock(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return _acquire_compile_lock

    def _is_owned(self):
        if self._lock.acquire(0):
            self._lock.release()
            return False
        else:
            return True