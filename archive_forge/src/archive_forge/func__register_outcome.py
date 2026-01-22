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
def _register_outcome(self, outcome):
    """Register the outcome of a task.

        This method can be called only once, future calls will be ignored.
        """
    with self.parallel._lock:
        if self.status not in (TASK_PENDING, None):
            return
        self.status = outcome['status']
    self._result = outcome['result']
    self.job = None
    if self.status == TASK_ERROR:
        self.parallel._exception = True
        self.parallel._aborting = True