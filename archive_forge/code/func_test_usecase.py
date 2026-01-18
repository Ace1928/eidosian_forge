import numpy as np
import unittest
from numba import njit
from numba.core import types, errors
from numba.tests.support import TestCase
def test_usecase(self):
    n = 10
    obs_got = np.zeros(n)
    obs_expected = obs_got.copy()
    cfunc = njit((types.float64[:], types.intp))(usecase)
    cfunc(obs_got, n)
    usecase(obs_expected, n)
    self.assertPreciseEqual(obs_got, obs_expected)