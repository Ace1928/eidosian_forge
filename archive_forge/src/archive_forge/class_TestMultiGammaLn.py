import numpy as np
from numpy.testing import (assert_array_equal,
from pytest import raises as assert_raises
from scipy.special import gammaln, multigammaln
class TestMultiGammaLn:

    def test1(self):
        np.random.seed(1234)
        a = np.abs(np.random.randn())
        assert_array_equal(multigammaln(a, 1), gammaln(a))

    def test2(self):
        a = np.array([2.5, 10.0])
        result = multigammaln(a, 2)
        expected = np.log(np.sqrt(np.pi)) + gammaln(a) + gammaln(a - 0.5)
        assert_almost_equal(result, expected)

    def test_bararg(self):
        assert_raises(ValueError, multigammaln, 0.5, 1.2)