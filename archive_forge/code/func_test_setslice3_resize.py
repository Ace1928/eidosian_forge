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
def test_setslice3_resize(self):
    self.disable_leak_check()
    pyfunc = list_setslice3_arbitrary
    cfunc = jit(nopython=True)(pyfunc)
    cfunc(5, 10, 0, 2, 1)
    with self.assertRaises(ValueError) as cm:
        cfunc(5, 100, 0, 3, 2)
    self.assertIn('cannot resize', str(cm.exception))