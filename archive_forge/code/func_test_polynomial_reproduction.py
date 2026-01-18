import pickle
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats.qmc import Halton
from scipy.spatial import cKDTree
from scipy.interpolate._rbfinterp import (
from scipy.interpolate import _rbfinterp_pythran
def test_polynomial_reproduction(self):
    rng = np.random.RandomState(0)
    seq = Halton(2, scramble=False, seed=rng)
    degree = 3
    x = seq.random(50)
    xitp = seq.random(50)
    P = _vandermonde(x, degree)
    Pitp = _vandermonde(xitp, degree)
    poly_coeffs = rng.normal(0.0, 1.0, P.shape[1])
    y = P.dot(poly_coeffs)
    yitp1 = Pitp.dot(poly_coeffs)
    yitp2 = self.build(x, y, degree=degree)(xitp)
    assert_allclose(yitp1, yitp2, atol=1e-08)