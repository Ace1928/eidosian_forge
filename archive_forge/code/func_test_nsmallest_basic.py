import heapq as hq
import itertools
import numpy as np
from numba import jit, typed
from numba.tests.support import TestCase, MemoryLeakMixin
def test_nsmallest_basic(self):
    pyfunc = nsmallest
    cfunc = jit(nopython=True)(pyfunc)
    for iterable in self.iterables():
        for n in range(-5, len(iterable) + 3):
            expected = pyfunc(1, iterable)
            got = cfunc(1, iterable)
            self.assertPreciseEqual(expected, got)
    out = cfunc(False, self.listimpl([3, 2, 1]))
    self.assertPreciseEqual(out, [])
    out = cfunc(True, self.listimpl([3, 2, 1]))
    self.assertPreciseEqual(out, [1])
    out = cfunc(2, (6, 5, 4, 3, 2, 1))
    self.assertPreciseEqual(out, [1, 2])
    out = cfunc(3, np.arange(6))
    self.assertPreciseEqual(out, [0, 1, 2])