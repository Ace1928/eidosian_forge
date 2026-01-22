import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
import scipy.special as sc
class TestInverseErrorFunction:

    def test_compliment(self):
        x = np.linspace(-1, 1, 101)
        assert_allclose(sc.erfcinv(1 - x), sc.erfinv(x), rtol=0, atol=1e-15)

    def test_literal_values(self):
        y = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        actual = sc.erfinv(y)
        expected = [0.0, 0.08885599049425769, 0.1791434546212917, 0.2724627147267543, 0.37080715859355795, 0.4769362762044699, 0.5951160814499948, 0.7328690779592167, 0.9061938024368233, 1.1630871536766743]
        assert_allclose(actual, expected, rtol=0, atol=1e-15)

    @pytest.mark.parametrize('f, x, y', [(sc.erfinv, -1, -np.inf), (sc.erfinv, 0, 0), (sc.erfinv, 1, np.inf), (sc.erfinv, -100, np.nan), (sc.erfinv, 100, np.nan), (sc.erfcinv, 0, np.inf), (sc.erfcinv, 1, -0.0), (sc.erfcinv, 2, -np.inf), (sc.erfcinv, -100, np.nan), (sc.erfcinv, 100, np.nan)], ids=['erfinv at lower bound', 'erfinv at midpoint', 'erfinv at upper bound', 'erfinv below lower bound', 'erfinv above upper bound', 'erfcinv at lower bound', 'erfcinv at midpoint', 'erfcinv at upper bound', 'erfcinv below lower bound', 'erfcinv above upper bound'])
    def test_domain_bounds(self, f, x, y):
        assert_equal(f(x), y)

    def test_erfinv_asympt(self):
        x = np.array([1e-20, 1e-15, 1e-14, 1e-10, 1e-08, 9e-08, 1.1e-07, 1e-06])
        expected = np.array([8.86226925452758e-21, 8.862269254527581e-16, 8.86226925452758e-15, 8.862269254527581e-11, 8.86226925452758e-09, 7.97604232907484e-08, 9.74849617998037e-08, 8.8622692545299e-07])
        assert_allclose(sc.erfinv(x), expected, rtol=1e-15)
        assert_allclose(sc.erf(sc.erfinv(x)), x, rtol=5e-15)