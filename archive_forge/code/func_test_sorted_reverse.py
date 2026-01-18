import copy
import itertools
import math
import random
import sys
import unittest
import numpy as np
from numba import jit, njit
from numba.core import utils, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.misc.quicksort import make_py_quicksort, make_jit_quicksort
from numba.misc.mergesort import make_jit_mergesort
from numba.misc.timsort import make_py_timsort, make_jit_timsort, MergeRun
def test_sorted_reverse(self):
    pyfunc = sorted_reverse_usecase
    cfunc = jit(nopython=True)(pyfunc)
    size = 20
    orig = np.random.random(size=size) * 100
    for b in (False, True):
        expected = sorted(orig, reverse=b)
        got = cfunc(orig, b)
        self.assertPreciseEqual(got, expected)
        self.assertNotEqual(list(orig), got)