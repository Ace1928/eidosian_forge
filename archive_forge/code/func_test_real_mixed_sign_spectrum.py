import random
import functools
import numpy as np
from numpy import array, identity, dot, sqrt
from numpy.testing import (assert_array_almost_equal, assert_allclose, assert_,
import pytest
import scipy.linalg
from scipy.linalg import (funm, signm, logm, sqrtm, fractional_matrix_power,
from scipy.linalg import _matfuncs_inv_ssq
import scipy.linalg._expm_frechet
from scipy.optimize import minimize
def test_real_mixed_sign_spectrum(self):
    for M in ([[1, 0], [0, -1]], [[0, 1], [1, 0]]):
        for dt in (float, complex):
            A = np.array(M, dtype=dt)
            A_logm, info = logm(A, disp=False)
            assert_(np.issubdtype(A_logm.dtype, np.complexfloating))