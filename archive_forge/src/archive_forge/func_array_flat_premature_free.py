import itertools
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
import unittest
def array_flat_premature_free(size):
    x = np.arange(size)
    res = np.zeros_like(x, dtype=np.intp)
    for i, v in enumerate(x.flat):
        res[i] = v
    return res