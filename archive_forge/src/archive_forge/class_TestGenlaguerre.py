import numpy as np
from numpy import array, sqrt
from numpy.testing import (assert_array_almost_equal, assert_equal,
from pytest import raises as assert_raises
from scipy import integrate
import scipy.special as sc
from scipy.special import gamma
import scipy.special._orthogonal as orth
class TestGenlaguerre:

    def test_regression(self):
        assert_equal(orth.genlaguerre(1, 1, monic=False)(0), 2.0)
        assert_equal(orth.genlaguerre(1, 1, monic=True)(0), -2.0)
        assert_equal(orth.genlaguerre(1, 1, monic=False), np.poly1d([-1, 2]))
        assert_equal(orth.genlaguerre(1, 1, monic=True), np.poly1d([1, -2]))