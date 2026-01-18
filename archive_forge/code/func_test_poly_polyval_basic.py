import gc
from itertools import product
import numpy as np
from numpy.polynomial import polynomial as poly
from numpy.polynomial import polyutils as pu
from numba import jit, njit
from numba.tests.support import (TestCase, needs_lapack,
from numba.core.errors import TypingError
def test_poly_polyval_basic(self):
    pyfunc2 = polyval2
    cfunc2 = njit(polyval2)
    pyfunc3T = polyval3T
    cfunc3T = njit(polyval3T)
    pyfunc3F = polyval3F
    cfunc3F = njit(polyval3F)

    def inputs():
        yield (np.array([], dtype=np.float64), [1])
        yield (1, [1, 2, 3])
        yield (np.arange(4).reshape(2, 2), [1, 2, 3])
        for i in range(5):
            yield (np.linspace(-1, 1), [0] * i + [1])
        yield (np.linspace(-1, 1), [0, -1, 0, 1])
        for i in range(3):
            dims = [2] * i
            x = np.zeros(dims)
            yield (x, [1])
            yield (x, [1, 0])
            yield (x, [1, 0, 0])
        yield (np.array([1, 2]), np.arange(4).reshape(2, 2))
        yield ([1, 2], np.arange(4).reshape(2, 2))
    for x, c in inputs():
        self.assertPreciseEqual(pyfunc2(x, c), cfunc2(x, c))
        self.assertPreciseEqual(pyfunc3T(x, c), cfunc3T(x, c))
        self.assertPreciseEqual(pyfunc3F(x, c), cfunc3F(x, c))