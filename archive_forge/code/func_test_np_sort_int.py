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
def test_np_sort_int(self):
    pyfunc = np_sort_usecase
    cfunc = jit(nopython=True)(pyfunc)
    for orig in self.int_arrays():
        self.check_sort_copy(pyfunc, cfunc, orig)