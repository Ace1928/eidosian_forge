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
def assertSortedValues(self, orig, orig_values, result, result_values):
    self.assertEqual(len(result), len(orig))
    self.assertEqual(list(result), sorted(orig))
    zip_sorted = sorted(zip(orig, orig_values), key=lambda x: x[0])
    zip_result = list(zip(result, result_values))
    self.assertEqual(zip_sorted, zip_result)
    for i in range(len(zip_result) - 1):
        (k1, v1), (k2, v2) = (zip_result[i], zip_result[i + 1])
        if k1 == k2:
            self.assertLess(orig_values.index(v1), orig_values.index(v2))