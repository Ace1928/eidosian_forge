import numpy as np
from numba import float32, jit, njit
from numba.np.ufunc import Vectorize
from numba.core.errors import TypingError
from numba.tests.support import TestCase
import unittest
def test_ufunc_attributes(self):
    for v in vectorizers:
        self._test_ufunc_attributes(v, a[0], b[0])
    for v in vectorizers:
        self._test_ufunc_attributes(v, a, b)
    for v in vectorizers:
        self._test_ufunc_attributes(v, a[:, np.newaxis, :], b[np.newaxis, :, :])