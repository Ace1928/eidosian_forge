import numpy as np
import pandas as pd
from numpy.testing import (assert_almost_equal, assert_raises, assert_equal,
from statsmodels.stats._adnorm import normal_ad
from statsmodels.stats.stattools import (omni_normtest, jarque_bera,
def test_robust_skewness_1d_2d(self, reset_randomstate):
    x = np.random.randn(21)
    y = x[:, None]
    sk_x = robust_skewness(x)
    sk_y = robust_skewness(y, axis=None)
    assert_almost_equal(np.array(sk_x), np.array(sk_y))