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
@with_multiprocessing
@parametrize('n_jobs', [2, -1])
@parametrize('var_name', ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'])
@parametrize('context', [parallel_config, parallel_backend])
def test_threadpool_limitation_in_child_override(context, n_jobs, var_name):
    get_reusable_executor(reuse=True).shutdown()

    def _get_env(var_name):
        return os.environ.get(var_name)
    original_var_value = os.environ.get(var_name)
    try:
        os.environ[var_name] = '4'
        results = Parallel(n_jobs=n_jobs)((delayed(_get_env)(var_name) for i in range(2)))
        assert results == ['4', '4']
        with context('loky', inner_max_num_threads=1):
            results = Parallel(n_jobs=n_jobs)((delayed(_get_env)(var_name) for i in range(2)))
        assert results == ['1', '1']
    finally:
        if original_var_value is None:
            del os.environ[var_name]
        else:
            os.environ[var_name] = original_var_value