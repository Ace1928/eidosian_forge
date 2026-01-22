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
def _get_theoretical_diff_between_simps_and_cum_simps(self, y, x):
    """`cumulative_simpson` and `simpson` can be tested against other to verify
        they give consistent results. `simpson` will iteratively be called with 
        successively higher upper limits of integration. This function calculates
        the theoretical correction required to `simpson` at even intervals to match
        with `cumulative_simpson`.
        """
    d = np.diff(x, axis=-1)
    sub_integrals_h1 = _cumulative_simpson_unequal_intervals(y, d)
    sub_integrals_h2 = _cumulative_simpson_unequal_intervals(y[..., ::-1], d[..., ::-1])[..., ::-1]
    zeros_shape = (*y.shape[:-1], 1)
    theoretical_difference = np.concatenate([np.zeros(zeros_shape), sub_integrals_h1[..., 1:] - sub_integrals_h2[..., :-1], np.zeros(zeros_shape)], axis=-1)
    theoretical_difference[..., 1::2] = 0.0
    return theoretical_difference