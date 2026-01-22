import pytest
import numpy as np
from numpy import cos, sin, pi
from numpy.testing import (assert_equal, assert_almost_equal, assert_allclose,
from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as hyp_num
from scipy.integrate import (quadrature, romberg, romb, newton_cotes,
from scipy.integrate._quadrature import _cumulative_simpson_unequal_intervals
from scipy.integrate._tanhsinh import _tanhsinh, _pair_cache
from scipy import stats, special as sc
from scipy.optimize._zeros_py import (_ECONVERGED, _ESIGNERR, _ECONVERR,  # noqa: F401
class TestFixedQuad:

    def test_scalar(self):
        n = 4
        expected = 1 / (2 * n)
        got, _ = fixed_quad(lambda x: x ** (2 * n - 1), 0, 1, n=n)
        assert_allclose(got, expected, rtol=1e-12)

    def test_vector(self):
        n = 4
        p = np.arange(1, 2 * n)
        expected = 1 / (p + 1)
        got, _ = fixed_quad(lambda x: x ** p[:, None], 0, 1, n=n)
        assert_allclose(got, expected, rtol=1e-12)