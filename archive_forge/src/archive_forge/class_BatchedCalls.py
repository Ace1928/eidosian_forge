from __future__ import division
import os
import sys
from math import sqrt
import functools
import collections
import time
import threading
import itertools
from uuid import uuid4
from numbers import Integral
import warnings
import queue
import weakref
from contextlib import nullcontext
from multiprocessing import TimeoutError
from ._multiprocessing_helpers import mp
from .logger import Logger, short_format_time
from .disk import memstr_to_bytes
from ._parallel_backends import (FallbackToBackend, MultiprocessingBackend,
from ._utils import eval_expr, _Sentinel
from ._parallel_backends import AutoBatchingMixin  # noqa
from ._parallel_backends import ParallelBackendBase  # noqa
class BatchedCalls(object):
    """Wrap a sequence of (func, args, kwargs) tuples as a single callable"""

    def __init__(self, iterator_slice, backend_and_jobs, reducer_callback=None, pickle_cache=None):
        self.items = list(iterator_slice)
        self._size = len(self.items)
        self._reducer_callback = reducer_callback
        if isinstance(backend_and_jobs, tuple):
            self._backend, self._n_jobs = backend_and_jobs
        else:
            self._backend, self._n_jobs = (backend_and_jobs, None)
        self._pickle_cache = pickle_cache if pickle_cache is not None else {}

    def __call__(self):
        with parallel_config(backend=self._backend, n_jobs=self._n_jobs):
            return [func(*args, **kwargs) for func, args, kwargs in self.items]

    def __reduce__(self):
        if self._reducer_callback is not None:
            self._reducer_callback()
        return (BatchedCalls, (self.items, (self._backend, self._n_jobs), None, self._pickle_cache))

    def __len__(self):
        return self._size