import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
from scipy.special import logit, expit, log_expit
class TestLogit:

    def check_logit_out(self, dtype, expected):
        a = np.linspace(0, 1, 10)
        a = np.array(a, dtype=dtype)
        with np.errstate(divide='ignore'):
            actual = logit(a)
        assert_almost_equal(actual, expected)
        assert_equal(actual.dtype, np.dtype(dtype))

    def test_float32(self):
        expected = np.array([-np.inf, -2.07944155, -1.25276291, -0.69314718, -0.22314353, 0.22314365, 0.6931473, 1.25276303, 2.07944155, np.inf], dtype=np.float32)
        self.check_logit_out('f4', expected)

    def test_float64(self):
        expected = np.array([-np.inf, -2.07944154, -1.25276297, -0.69314718, -0.22314355, 0.22314355, 0.69314718, 1.25276297, 2.07944154, np.inf])
        self.check_logit_out('f8', expected)

    def test_nan(self):
        expected = np.array([np.nan] * 4)
        with np.errstate(invalid='ignore'):
            actual = logit(np.array([-3.0, -2.0, 2.0, 3.0]))
        assert_equal(expected, actual)