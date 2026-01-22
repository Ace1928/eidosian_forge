from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def _set_some_values_to_nan(a):
    p = a.size // 2
    np.put(a, np.random.choice(range(a.size), p, replace=False), np.nan)
    return a