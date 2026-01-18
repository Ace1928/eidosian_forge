import numpy as np
import pandas as pd
from numpy.testing import (assert_almost_equal, assert_raises, assert_equal,
from statsmodels.stats._adnorm import normal_ad
from statsmodels.stats.stattools import (omni_normtest, jarque_bera,
def test_robust_skewness_symmetric(self, reset_randomstate):
    x = np.random.standard_normal(100)
    x = np.hstack([x, np.zeros(1), -x])
    sk = robust_skewness(x)
    assert_almost_equal(np.array(sk), np.zeros(4))