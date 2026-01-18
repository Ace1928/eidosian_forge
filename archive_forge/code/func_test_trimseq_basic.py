import gc
from itertools import product
import numpy as np
from numpy.polynomial import polynomial as poly
from numpy.polynomial import polyutils as pu
from numba import jit, njit
from numba.tests.support import (TestCase, needs_lapack,
from numba.core.errors import TypingError
def test_trimseq_basic(self):
    pyfunc = trimseq
    cfunc = njit(trimseq)

    def inputs():
        for i in range(5):
            yield np.array([1] + [0] * i)
    for coefs in inputs():
        self.assertPreciseEqual(pyfunc(coefs), cfunc(coefs))