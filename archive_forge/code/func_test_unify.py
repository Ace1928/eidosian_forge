from collections import namedtuple
import contextlib
import itertools
import math
import sys
import ctypes as ct
import numpy as np
from numba import jit, typeof, njit, literal_unroll, literally
import unittest
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.experimental import jitclass
from numba.core.extending import overload
def test_unify(self):

    @njit
    def foo(x):
        if x + 1 > 3:
            l = ['a', 1]
        else:
            l = ['b', 2]
        return l[0]
    for x in (-100, 100):
        self.assertEqual(foo.py_func(x), foo(x))