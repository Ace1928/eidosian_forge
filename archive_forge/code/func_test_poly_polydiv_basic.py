import gc
from itertools import product
import numpy as np
from numpy.polynomial import polynomial as poly
from numpy.polynomial import polyutils as pu
from numba import jit, njit
from numba.tests.support import (TestCase, needs_lapack,
from numba.core.errors import TypingError
def test_poly_polydiv_basic(self):
    pyfunc = polydiv
    cfunc = njit(polydiv)
    self._test_polyarithm_basic(polydiv)

    def inputs():
        yield ([2], [2])
        yield ([2, 2], [2])
        for i in range(5):
            for j in range(5):
                ci = [0] * i + [1, 2]
                cj = [0] * j + [1, 2]
                tgt = poly.polyadd(ci, cj)
                yield (tgt, ci)
        yield (np.array([1, 0, 0, 0, 0, 0, -1]), np.array([1, 0, 0, -1]))
    for c1, c2 in inputs():
        self.assertPreciseEqual(pyfunc(c1, c2), cfunc(c1, c2))