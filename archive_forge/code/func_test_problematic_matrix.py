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
def test_problematic_matrix(self):
    A = np.array([[1.50591997, 1.93537998], [0.41203263, 0.23443516]], dtype=float)
    E = np.array([[1.87864034, 2.07055038], [1.34102727, 0.67341123]], dtype=float)
    scipy.linalg.norm(A, 1)
    sps_expm, sps_frechet = expm_frechet(A, E, method='SPS')
    blockEnlarge_expm, blockEnlarge_frechet = expm_frechet(A, E, method='blockEnlarge')
    assert_allclose(sps_expm, blockEnlarge_expm)
    assert_allclose(sps_frechet, blockEnlarge_frechet)