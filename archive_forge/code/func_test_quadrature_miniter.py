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
def test_quadrature_miniter(self):

    def myfunc(x, n, z):
        return cos(n * x - z * sin(x)) / pi
    table_val = 0.30614353532540295
    for miniter in [5, 52]:
        val, err = quadrature(myfunc, 0, pi, (2, 1.8), miniter=miniter)
        assert_almost_equal(val, table_val, decimal=7)
        assert_(err < 1.0)