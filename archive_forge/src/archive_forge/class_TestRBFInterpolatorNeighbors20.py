import pickle
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats.qmc import Halton
from scipy.spatial import cKDTree
from scipy.interpolate._rbfinterp import (
from scipy.interpolate import _rbfinterp_pythran
class TestRBFInterpolatorNeighbors20(_TestRBFInterpolator):

    def build(self, *args, **kwargs):
        return RBFInterpolator(*args, **kwargs, neighbors=20)

    def test_equivalent_to_rbf_interpolator(self):
        seq = Halton(2, scramble=False, seed=np.random.RandomState())
        x = seq.random(100)
        xitp = seq.random(100)
        y = _2d_test_function(x)
        yitp1 = self.build(x, y)(xitp)
        yitp2 = []
        tree = cKDTree(x)
        for xi in xitp:
            _, nbr = tree.query(xi, 20)
            yitp2.append(RBFInterpolator(x[nbr], y[nbr])(xi[None])[0])
        assert_allclose(yitp1, yitp2, atol=1e-08)