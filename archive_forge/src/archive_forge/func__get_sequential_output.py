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
def _get_sequential_output(self, iterable):
    """Separate loop for sequential output.

        This simplifies the traceback in case of errors and reduces the
        overhead of calling sequential tasks with `joblib`.
        """
    try:
        self._iterating = True
        self._original_iterator = iterable
        batch_size = self._get_batch_size()
        if batch_size != 1:
            it = iter(iterable)
            iterable_batched = iter(lambda: tuple(itertools.islice(it, batch_size)), ())
            iterable = (task for batch in iterable_batched for task in batch)
        yield None
        for func, args, kwargs in iterable:
            self.n_dispatched_batches += 1
            self.n_dispatched_tasks += 1
            res = func(*args, **kwargs)
            self.n_completed_tasks += 1
            self.print_progress()
            yield res
            self._nb_consumed += 1
    except BaseException:
        self._exception = True
        self._aborting = True
        self._aborted = True
        raise
    finally:
        self.print_progress()
        self._running = False
        self._iterating = False
        self._original_iterator = None