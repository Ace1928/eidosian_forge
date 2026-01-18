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
def test_staticgetitem(self):

    @njit
    def foo():
        l = ['a', 1]
        return (l[0], l[1])
    self.assertEqual(foo.py_func(), foo())