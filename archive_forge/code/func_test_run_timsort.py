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
def test_run_timsort(self):
    f = self.timsort.run_timsort
    for size_factor in (1, 10):
        sizes = (15, 30, 20)
        all_lists = [self.make_sample_lists(n * size_factor) for n in sizes]
        for chunks in itertools.product(*all_lists):
            orig_keys = sum(chunks, [])
            keys = self.array_factory(orig_keys)
            f(keys)
            self.assertSorted(orig_keys, keys)