import itertools
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
import unittest
def test_np_ndindex_empty(self):
    func = np_ndindex_empty
    cfunc = njit(())(func)
    self.assertPreciseEqual(cfunc(), func())