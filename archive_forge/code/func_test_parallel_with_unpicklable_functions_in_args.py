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
@parametrize('backend', PROCESS_BACKENDS + ([] if mp is None else ['spawn']))
@parametrize('define_func', [SQUARE_MAIN, SQUARE_LOCAL, SQUARE_LAMBDA])
@parametrize('callable_position', ['delayed', 'args', 'kwargs'])
def test_parallel_with_unpicklable_functions_in_args(backend, define_func, callable_position, tmpdir):
    if backend in ['multiprocessing', 'spawn'] and (define_func != SQUARE_MAIN or sys.platform == 'win32'):
        pytest.skip('Not picklable with pickle')
    code = UNPICKLABLE_CALLABLE_SCRIPT_TEMPLATE_MAIN.format(define_func=define_func, backend=backend, callable_position=callable_position, joblib_root_folder=os.path.dirname(os.path.dirname(joblib.__file__)))
    code_file = tmpdir.join('unpicklable_func_script.py')
    code_file.write(code)
    check_subprocess_call([sys.executable, code_file.strpath], timeout=10, stdout_regex='\\[0, 1, 4, 9, 16\\]')