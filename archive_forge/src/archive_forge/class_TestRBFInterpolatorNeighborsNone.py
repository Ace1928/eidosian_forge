import pickle
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats.qmc import Halton
from scipy.spatial import cKDTree
from scipy.interpolate._rbfinterp import (
from scipy.interpolate import _rbfinterp_pythran
class TestRBFInterpolatorNeighborsNone(_TestRBFInterpolator):

    def build(self, *args, **kwargs):
        return RBFInterpolator(*args, **kwargs)

    def test_smoothing_limit_1d(self):
        seq = Halton(1, scramble=False, seed=np.random.RandomState())
        degree = 3
        smoothing = 100000000.0
        x = 3 * seq.random(50)
        xitp = 3 * seq.random(50)
        y = _1d_test_function(x)
        yitp1 = self.build(x, y, degree=degree, smoothing=smoothing)(xitp)
        P = _vandermonde(x, degree)
        Pitp = _vandermonde(xitp, degree)
        yitp2 = Pitp.dot(np.linalg.lstsq(P, y, rcond=None)[0])
        assert_allclose(yitp1, yitp2, atol=1e-08)

    def test_smoothing_limit_2d(self):
        seq = Halton(2, scramble=False, seed=np.random.RandomState())
        degree = 3
        smoothing = 100000000.0
        x = seq.random(100)
        xitp = seq.random(100)
        y = _2d_test_function(x)
        yitp1 = self.build(x, y, degree=degree, smoothing=smoothing)(xitp)
        P = _vandermonde(x, degree)
        Pitp = _vandermonde(xitp, degree)
        yitp2 = Pitp.dot(np.linalg.lstsq(P, y, rcond=None)[0])
        assert_allclose(yitp1, yitp2, atol=1e-08)