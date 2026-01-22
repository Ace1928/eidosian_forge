from concurrent import futures
import logging
class _LoggingPool(object):
    """An exception-logging futures.ThreadPoolExecutor-compatible thread pool."""

    def __init__(self, backing_pool):
        self._backing_pool = backing_pool

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._backing_pool.shutdown(wait=True)

    def submit(self, fn, *args, **kwargs):
        return self._backing_pool.submit(_wrap(fn), *args, **kwargs)

    def map(self, func, *iterables, **kwargs):
        return self._backing_pool.map(_wrap(func), *iterables, timeout=kwargs.get('timeout', None))

    def shutdown(self, wait=True):
        self._backing_pool.shutdown(wait=wait)