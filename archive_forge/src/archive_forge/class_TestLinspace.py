import contextlib
import sys
import numpy as np
import random
import re
import threading
import gc
from numba.core.errors import TypingError
from numba import njit
from numba.core import types, utils, config
from numba.tests.support import MemoryLeakMixin, TestCase, tag, skip_if_32bit
import unittest
class TestLinspace(BaseTest):

    def test_linspace_2(self):

        def pyfunc(n, m):
            return np.linspace(n, m)
        self.check_outputs(pyfunc, [(0, 4), (1, 100), (-3.5, 2.5), (-3j, 2 + 3j), (2, 1), (1 + 0.5j, 1.5j)])

    def test_linspace_3(self):

        def pyfunc(n, m, p):
            return np.linspace(n, m, p)
        self.check_outputs(pyfunc, [(0, 4, 9), (1, 4, 3), (-3.5, 2.5, 8), (-3j, 2 + 3j, 7), (2, 1, 0), (1 + 0.5j, 1.5j, 5), (1, 1e+100, 1)])

    def test_linspace_accuracy(self):

        @nrtjit
        def foo(n, m, p):
            return np.linspace(n, m, p)
        n, m, p = (0.0, 1.0, 100)
        self.assertPreciseEqual(foo(n, m, p), foo.py_func(n, m, p))