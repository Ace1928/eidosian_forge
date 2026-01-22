import gc
import os
import warnings
import threading
import contextlib
from abc import ABCMeta, abstractmethod
from ._multiprocessing_helpers import mp
def _get_pool(self):
    """Lazily initialize the thread pool

        The actual pool of worker threads is only initialized at the first
        call to apply_async.
        """
    if self._pool is None:
        self._pool = ThreadPool(self._n_jobs)
    return self._pool