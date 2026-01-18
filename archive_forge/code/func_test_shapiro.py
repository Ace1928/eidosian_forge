import numpy as np
import pandas as pd
from numpy.testing import (assert_almost_equal, assert_raises, assert_equal,
from statsmodels.stats._adnorm import normal_ad
from statsmodels.stats.stattools import (omni_normtest, jarque_bera,
def test_shapiro():
    from scipy.stats import shapiro
    st_pv_R = np.array([0.939984787255526, 0.23962189800046])
    sh = shapiro(x)
    assert_almost_equal(sh, st_pv_R, 4)
    st_pv_R = np.array([0.5799574255943298, 1.838456834681376e-06 * 10000.0])
    sh = shapiro(x ** 2) * np.array([1, 10000.0])
    assert_almost_equal(sh, st_pv_R, 5)
    st_pv_R = np.array([0.9173044264316559, 0.08793704167882448])
    sh = shapiro(np.log(x ** 2))
    assert_almost_equal(sh, st_pv_R, 5)
    st_pv_R = np.array([0.8183618634939194, 0.001644620895206969])
    sh = shapiro(np.exp(-x ** 2))
    assert_almost_equal(sh, st_pv_R, 5)