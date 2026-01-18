import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
import statsmodels.stats.proportion as smprop
from statsmodels.stats.proportion import (
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
from statsmodels.tools.testing import Holder
from statsmodels.stats.tests.results.results_proportion import res_binom, res_binom_methods
def test_power_ztost_prop_norm():
    with pytest.warns(HypothesisTestWarning):
        power = smprop.power_ztost_prop(0.4, 0.6, np.arange(20, 210, 20), p_alt=0.5, alpha=0.05, discrete=False, dist='norm', variance_prop=0.5, continuity=0, critval_continuity=0)[0]
    res_power = np.array([0.0, 0.0, 0.0, 0.11450013, 0.27752006, 0.41495922, 0.52944621, 0.62382638, 0.70092914, 0.76341806])
    assert_almost_equal(np.maximum(power, 0), res_power, decimal=4)
    with pytest.warns(HypothesisTestWarning):
        power = smprop.power_ztost_prop(0.4, 0.6, np.arange(20, 210, 20), p_alt=0.5, alpha=0.05, discrete=False, dist='norm', variance_prop=0.5, continuity=1, critval_continuity=0)[0]
    res_power = np.array([0.0, 0.0, 0.02667562, 0.20189793, 0.35099606, 0.47608598, 0.57981118, 0.66496683, 0.73427591, 0.79026127])
    assert_almost_equal(np.maximum(power, 0), res_power, decimal=4)
    with pytest.warns(HypothesisTestWarning):
        power = smprop.power_ztost_prop(0.4, 0.6, np.arange(20, 210, 20), p_alt=0.5, alpha=0.05, discrete=True, dist='norm', variance_prop=0.5, continuity=1, critval_continuity=0)[0]
    res_power = np.array([0.0, 0.0, 0.0, 0.08902071, 0.23582284, 0.35192313, 0.55312718, 0.61549537, 0.66743625, 0.77066806])
    assert_almost_equal(np.maximum(power, 0), res_power, decimal=4)
    with pytest.warns(HypothesisTestWarning):
        power = smprop.power_ztost_prop(0.4, 0.6, np.arange(20, 210, 20), p_alt=0.5, alpha=0.05, discrete=True, dist='norm', variance_prop=0.5, continuity=1, critval_continuity=1)[0]
    res_power = np.array([0.0, 0.0, 0.0, 0.08902071, 0.23582284, 0.35192313, 0.44588687, 0.61549537, 0.66743625, 0.71115563])
    assert_almost_equal(np.maximum(power, 0), res_power, decimal=4)
    with pytest.warns(HypothesisTestWarning):
        power = smprop.power_ztost_prop(0.4, 0.6, np.arange(20, 210, 20), p_alt=0.5, alpha=0.05, discrete=True, dist='norm', variance_prop=None, continuity=0, critval_continuity=0)[0]
    res_power = np.array([0.0, 0.0, 0.0, 0.0, 0.15851942, 0.41611758, 0.5010377, 0.5708047, 0.70328247, 0.74210096])
    assert_almost_equal(np.maximum(power, 0), res_power, decimal=4)