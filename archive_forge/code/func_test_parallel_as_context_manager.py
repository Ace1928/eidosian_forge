import os
import sys
import time
import mmap
import weakref
import warnings
import threading
from traceback import format_exception
from math import sqrt
from time import sleep
from pickle import PicklingError
from contextlib import nullcontext
from multiprocessing import TimeoutError
import pytest
import joblib
from joblib import parallel
from joblib import dump, load
from joblib._multiprocessing_helpers import mp
from joblib.test.common import np, with_numpy
from joblib.test.common import with_multiprocessing
from joblib.test.common import IS_PYPY, force_gc_pypy
from joblib.testing import (parametrize, raises, check_subprocess_call,
from queue import Queue
from joblib._parallel_backends import SequentialBackend
from joblib._parallel_backends import ThreadingBackend
from joblib._parallel_backends import MultiprocessingBackend
from joblib._parallel_backends import ParallelBackendBase
from joblib._parallel_backends import LokyBackend
from joblib.parallel import Parallel, delayed
from joblib.parallel import parallel_config
from joblib.parallel import parallel_backend
from joblib.parallel import register_parallel_backend
from joblib.parallel import effective_n_jobs, cpu_count
from joblib.parallel import mp, BACKENDS, DEFAULT_BACKEND
from joblib import Parallel, delayed
import sys
from joblib import Parallel, delayed
import sys
import faulthandler
from joblib import Parallel, delayed
from functools import partial
import sys
from joblib import Parallel, delayed, hash
import multiprocessing as mp
@parametrize('backend', PARALLEL_BACKENDS)
def test_parallel_as_context_manager(backend):
    lst = range(10)
    expected = [f(x, y=1) for x in lst]
    with Parallel(n_jobs=4, backend=backend) as p:
        managed_backend = p._backend
        assert expected == p((delayed(f)(x, y=1) for x in lst))
        assert expected == p((delayed(f)(x, y=1) for x in lst))
        if mp is not None:
            assert get_workers(managed_backend) is get_workers(p._backend)
    if mp is not None:
        assert get_workers(p._backend) is None
    assert expected == p((delayed(f)(x, y=1) for x in lst))
    if mp is not None:
        assert get_workers(p._backend) is None