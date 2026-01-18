import multiprocessing
import platform
import threading
import pickle
import weakref
from itertools import chain
from io import StringIO
import numpy as np
from numba import njit, jit, typeof, vectorize
from numba.core import types, errors
from numba import _dispatcher
from numba.tests.support import TestCase, captured_stdout
from numba.np.numpy_support import as_dtype
from numba.core.dispatcher import Dispatcher
from numba.extending import overload
from numba.tests.support import needs_lapack, SerialMixin
from numba.testing.main import _TIMEOUT as _RUNNER_TIMEOUT
import unittest
def test_pass_dispatcher_as_arg(self):

    @jit(nopython=True)
    def add1(x):
        return x + 1

    @jit(nopython=True)
    def bar(fn, x):
        return fn(x)

    @jit(nopython=True)
    def foo(x):
        return bar(add1, x)
    inputs = [1, 11.1, np.arange(10)]
    expected_results = [x + 1 for x in inputs]
    for arg, expect in zip(inputs, expected_results):
        self.assertPreciseEqual(foo(arg), expect)
    for arg, expect in zip(inputs, expected_results):
        self.assertPreciseEqual(bar(add1, arg), expect)