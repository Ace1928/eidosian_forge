import gc
from itertools import product
import numpy as np
from numpy.polynomial import polynomial as poly
from numpy.polynomial import polyutils as pu
from numba import jit, njit
from numba.tests.support import (TestCase, needs_lapack,
from numba.core.errors import TypingError
def test_poly_polydiv_exception(self):
    self._test_polyarithm_exception(polydiv)
    cfunc = njit(polydiv)
    with self.assertRaises(ZeroDivisionError) as _:
        cfunc([1], [0])