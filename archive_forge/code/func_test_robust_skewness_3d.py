import numpy as np
import pandas as pd
from numpy.testing import (assert_almost_equal, assert_raises, assert_equal,
from statsmodels.stats._adnorm import normal_ad
from statsmodels.stats.stattools import (omni_normtest, jarque_bera,
def test_robust_skewness_3d(self, reset_randomstate):
    x = np.random.standard_normal(100)
    x = np.hstack([x, np.zeros(1), -x])
    x = np.tile(x, (10, 10, 1))
    sk_3d = robust_skewness(x, axis=2)
    result = np.zeros((10, 10))
    for sk in sk_3d:
        assert_almost_equal(sk, result)