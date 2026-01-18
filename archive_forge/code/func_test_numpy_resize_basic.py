from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_numpy_resize_basic(self):
    pyfunc = numpy_resize
    cfunc = njit(pyfunc)

    def inputs():
        yield (np.array([[1, 2], [3, 4]]), (2, 4))
        yield (np.array([[1, 2], [3, 4]]), (4, 2))
        yield (np.array([[1, 2], [3, 4]]), (4, 3))
        yield (np.array([[1, 2], [3, 4]]), (0,))
        yield (np.array([[1, 2], [3, 4]]), (0, 2))
        yield (np.array([[1, 2], [3, 4]]), (2, 0))
        yield (np.zeros(0, dtype=float), (2, 1))
        yield (np.array([[1, 2], [3, 4]]), (4,))
        yield (np.array([[1, 2], [3, 4]]), 4)
        yield (np.zeros((1, 3), dtype=int), (2, 1))
        yield (np.array([], dtype=float), (4, 2))
        yield ([0, 1, 2, 3], (2, 3))
        yield (4, (2, 3))
    for a, new_shape in inputs():
        self.assertPreciseEqual(pyfunc(a, new_shape), cfunc(a, new_shape))