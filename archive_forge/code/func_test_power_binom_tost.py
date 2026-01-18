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
def test_power_binom_tost():
    p_alt = 0.6 + np.linspace(0, 0.09, 10)
    power = smprop.power_binom_tost(0.5, 0.7, 500, p_alt=p_alt, alpha=0.05)
    res_power = np.array([0.9965, 0.994, 0.9815, 0.9482, 0.8783, 0.7583, 0.5914, 0.4041, 0.2352, 0.1139])
    assert_almost_equal(power, res_power, decimal=4)
    rej_int = smprop.binom_tost_reject_interval(0.5, 0.7, 500)
    res_rej_int = (269, 332)
    assert_equal(rej_int, res_rej_int)
    nobs = np.arange(20, 210, 20)
    power = smprop.power_binom_tost(0.4, 0.6, nobs, p_alt=0.5, alpha=0.05)
    res_power = np.array([0.0, 0.0, 0.0, 0.0889, 0.2356, 0.3517, 0.4457, 0.6154, 0.6674, 0.7708])
    assert_almost_equal(np.maximum(power, 0), res_power, decimal=4)