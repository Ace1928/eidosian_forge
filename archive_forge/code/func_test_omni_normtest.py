import numpy as np
import pandas as pd
from numpy.testing import (assert_almost_equal, assert_raises, assert_equal,
from statsmodels.stats._adnorm import normal_ad
from statsmodels.stats.stattools import (omni_normtest, jarque_bera,
def test_omni_normtest():
    from scipy import stats
    st_pv_R = np.array([[3.994138321207883, -1.12930430216146, 1.648881473704978], [0.1357325110375005, 0.2587694866795507, 0.0991719192710234]])
    nt = omni_normtest(x)
    assert_almost_equal(nt, st_pv_R[:, 0], 14)
    st = stats.skewtest(x)
    assert_almost_equal(st, st_pv_R[:, 1], 14)
    kt = stats.kurtosistest(x)
    assert_almost_equal(kt, st_pv_R[:, 2], 11)
    st_pv_R = np.array([[34.523210399523926, 4.429509162503833, 3.860396220444025], [3.186985686465249e-08, 9.444780064482572e-06, 0.0001132033129378485]])
    x2 = x ** 2
    nt = omni_normtest(x2)
    assert_almost_equal(nt, st_pv_R[:, 0], 12)
    st = stats.skewtest(x2)
    assert_almost_equal(st, st_pv_R[:, 1], 12)
    kt = stats.kurtosistest(x2)
    assert_almost_equal(kt, st_pv_R[:, 2], 12)