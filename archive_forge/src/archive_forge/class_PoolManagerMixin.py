import gc
import os
import warnings
import threading
import contextlib
from abc import ABCMeta, abstractmethod
from ._multiprocessing_helpers import mp
class PoolManagerMixin(object):
    """A helper class for managing pool of workers."""
    _pool = None

    def effective_n_jobs(self, n_jobs):
        """Determine the number of jobs which are going to run in parallel"""
        if n_jobs == 0:
            raise ValueError('n_jobs == 0 in Parallel has no meaning')
        elif mp is None or n_jobs is None:
            return 1
        elif n_jobs < 0:
            n_jobs = max(cpu_count() + 1 + n_jobs, 1)
        return n_jobs

    def terminate(self):
        """Shutdown the process or thread pool"""
        if self._pool is not None:
            self._pool.close()
            self._pool.terminate()
            self._pool = None

    def _get_pool(self):
        """Used by apply_async to make it possible to implement lazy init"""
        return self._pool

    @staticmethod
    def _wrap_func_call(func):
        """Protect function call and return error with traceback."""
        try:
            return func()
        except BaseException as e:
            return _ExceptionWithTraceback(e)

    def apply_async(self, func, callback=None):
        """Schedule a func to be run"""
        return self._get_pool().apply_async(self._wrap_func_call, (func,), callback=callback, error_callback=callback)

    def retrieve_result_callback(self, out):
        """Mimic concurrent.futures results, raising an error if needed."""
        if isinstance(out, _ExceptionWithTraceback):
            rebuild, args = out.__reduce__()
            out = rebuild(*args)
        if isinstance(out, BaseException):
            raise out
        return out

    def abort_everything(self, ensure_ready=True):
        """Shutdown the pool and restart a new one with the same parameters"""
        self.terminate()
        if ensure_ready:
            self.configure(n_jobs=self.parallel.n_jobs, parallel=self.parallel, **self.parallel._backend_args)