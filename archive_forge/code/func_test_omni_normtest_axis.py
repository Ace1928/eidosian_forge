import numpy as np
import pandas as pd
from numpy.testing import (assert_almost_equal, assert_raises, assert_equal,
from statsmodels.stats._adnorm import normal_ad
from statsmodels.stats.stattools import (omni_normtest, jarque_bera,
def test_omni_normtest_axis(reset_randomstate):
    x = np.random.randn(25, 3)
    nt1 = omni_normtest(x)
    nt2 = omni_normtest(x, axis=0)
    nt3 = omni_normtest(x.T, axis=1)
    assert_almost_equal(nt2, nt1, decimal=13)
    assert_almost_equal(nt3, nt1, decimal=13)