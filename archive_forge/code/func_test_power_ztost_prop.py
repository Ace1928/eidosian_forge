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
def test_power_ztost_prop():
    power = smprop.power_ztost_prop(0.1, 0.9, 10, p_alt=0.6, alpha=0.05, discrete=True, dist='binom')[0]
    assert_almost_equal(power, 0.8204, decimal=4)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', HypothesisTestWarning)
        power = smprop.power_ztost_prop(0.4, 0.6, np.arange(20, 210, 20), p_alt=0.5, alpha=0.05, discrete=False, dist='binom')[0]
        res_power = np.array([0.0, 0.0, 0.0, 0.0889, 0.2356, 0.477, 0.553, 0.6154, 0.7365, 0.7708])
        assert_almost_equal(np.maximum(power, 0), res_power, decimal=4)
        power = smprop.power_ztost_prop(0.4, 0.6, np.arange(20, 210, 20), p_alt=0.5, alpha=0.05, discrete=False, dist='binom', variance_prop=None, continuity=2, critval_continuity=1)[0]
        res_power = np.array([0.0, 0.0, 0.0, 0.0889, 0.2356, 0.3517, 0.4457, 0.6154, 0.6674, 0.7708])
        assert_almost_equal(np.maximum(power, 0), res_power, decimal=4)
        power = smprop.power_ztost_prop(0.4, 0.6, np.arange(20, 210, 20), p_alt=0.5, alpha=0.05, discrete=False, dist='binom', variance_prop=0.5, critval_continuity=1)[0]
        res_power = np.array([0.0, 0.0, 0.0, 0.0889, 0.2356, 0.3517, 0.4457, 0.6154, 0.6674, 0.7112])
        assert_almost_equal(np.maximum(power, 0), res_power, decimal=4)