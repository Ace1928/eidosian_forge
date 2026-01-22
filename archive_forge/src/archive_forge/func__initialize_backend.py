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
def _initialize_backend(self):
    """Build a process or thread pool and return the number of workers"""
    try:
        n_jobs = self._backend.configure(n_jobs=self.n_jobs, parallel=self, **self._backend_args)
        if self.timeout is not None and (not self._backend.supports_timeout):
            warnings.warn("The backend class {!r} does not support timeout. You have set 'timeout={}' in Parallel but the 'timeout' parameter will not be used.".format(self._backend.__class__.__name__, self.timeout))
    except FallbackToBackend as e:
        self._backend = e.backend
        n_jobs = self._initialize_backend()
    return n_jobs