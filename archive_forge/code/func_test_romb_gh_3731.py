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
def test_romb_gh_3731(self):
    x = np.arange(2 ** 4 + 1)
    y = np.cos(0.2 * x)
    val = romb(y)
    val2, err = quad(lambda x: np.cos(0.2 * x), x.min(), x.max())
    assert_allclose(val, val2, rtol=1e-08, atol=0)
    with suppress_warnings() as sup:
        sup.filter(AccuracyWarning, 'divmax .4. exceeded')
        val3 = romberg(lambda x: np.cos(0.2 * x), x.min(), x.max(), divmax=4)
    assert_allclose(val, val3, rtol=1e-12, atol=0)