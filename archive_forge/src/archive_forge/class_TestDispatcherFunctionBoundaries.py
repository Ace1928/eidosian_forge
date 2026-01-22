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
class TestDispatcherFunctionBoundaries(TestCase):

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

    def test_dispatcher_as_arg_usecase(self):

        @jit(nopython=True)
        def maximum(seq, cmpfn):
            tmp = seq[0]
            for each in seq[1:]:
                cmpval = cmpfn(tmp, each)
                if cmpval < 0:
                    tmp = each
            return tmp
        got = maximum([1, 2, 3, 4], cmpfn=jit(lambda x, y: x - y))
        self.assertEqual(got, 4)
        got = maximum(list(zip(range(5), range(5)[::-1])), cmpfn=jit(lambda x, y: x[0] - y[0]))
        self.assertEqual(got, (4, 0))
        got = maximum(list(zip(range(5), range(5)[::-1])), cmpfn=jit(lambda x, y: x[1] - y[1]))
        self.assertEqual(got, (0, 4))

    def test_dispatcher_can_return_to_python(self):

        @jit(nopython=True)
        def foo(fn):
            return fn
        fn = jit(lambda x: x)
        self.assertEqual(foo(fn), fn)

    def test_dispatcher_in_sequence_arg(self):

        @jit(nopython=True)
        def one(x):
            return x + 1

        @jit(nopython=True)
        def two(x):
            return one(one(x))

        @jit(nopython=True)
        def three(x):
            return one(one(one(x)))

        @jit(nopython=True)
        def choose(fns, x):
            return (fns[0](x), fns[1](x), fns[2](x))
        self.assertEqual(choose((one, two, three), 1), (2, 3, 4))
        self.assertEqual(choose([one, one, one], 1), (2, 2, 2))