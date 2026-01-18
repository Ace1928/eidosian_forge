import copy
import warnings
import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose, assert_raises,
import pytest
import statsmodels.stats.power as smp
from statsmodels.stats.tests.test_weightstats import Holder
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
def test_normal_power_explicit():
    sigma = 1
    d = 0.3
    nobs = 80
    alpha = 0.05
    res1 = smp.normal_power(d, nobs / 2.0, 0.05)
    res2 = smp.NormalIndPower().power(d, nobs, 0.05)
    res3 = smp.NormalIndPower().solve_power(effect_size=0.3, nobs1=80, alpha=0.05, power=None)
    res_R = 0.475100870572638
    assert_almost_equal(res1, res_R, decimal=13)
    assert_almost_equal(res2, res_R, decimal=13)
    assert_almost_equal(res3, res_R, decimal=13)
    norm_pow = smp.normal_power(-0.01, nobs / 2.0, 0.05)
    norm_pow_R = 0.05045832927039234
    assert_almost_equal(norm_pow, norm_pow_R, decimal=11)
    norm_pow = smp.NormalIndPower().power(0.01, nobs, 0.05, alternative='larger')
    norm_pow_R = 0.056869534873146124
    assert_almost_equal(norm_pow, norm_pow_R, decimal=11)
    norm_pow = smp.NormalIndPower().power(-0.01, nobs, 0.05, alternative='larger')
    norm_pow_R = 0.0438089705093578
    assert_almost_equal(norm_pow, norm_pow_R, decimal=11)