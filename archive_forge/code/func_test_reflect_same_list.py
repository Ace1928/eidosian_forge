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
def test_reflect_same_list(self):
    """
        When the same list object is reflected twice, behaviour should
        be consistent.
        """
    pyfunc = reflect_dual
    cfunc = jit(nopython=True)(pyfunc)
    pylist = [1, 2, 3]
    clist = pylist[:]
    expected = pyfunc(pylist, pylist)
    got = cfunc(clist, clist)
    self.assertPreciseEqual(expected, got)
    self.assertPreciseEqual(pylist, clist)
    self.assertRefCountEqual(pylist, clist)