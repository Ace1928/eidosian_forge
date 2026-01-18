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
def test_partition3(self):
    n = 20

    def check(l, n):
        res = self.array_factory([9999] + l + [-9999])
        lt, gt = f(res, 1, n)
        self.assertEqual(res[0], 9999)
        self.assertEqual(res[-1], -9999)
        pivot = res[lt]
        for i in range(1, lt):
            self.assertLessEqual(res[i], pivot)
        for i in range(lt, gt + 1):
            self.assertEqual(res[i], pivot)
        for i in range(gt + 1, n):
            self.assertGreater(res[i], pivot)
    f = self.quicksort.partition3
    l = self.sorted_list(n)
    check(l, n)
    l = self.revsorted_list(n)
    check(l, n)
    l = self.initially_sorted_list(n, n // 2)
    check(l, n)
    l = self.revsorted_list(n)
    check(l, n)
    l = self.random_list(n)
    check(l, n)
    l = self.duprandom_list(n)
    check(l, n)