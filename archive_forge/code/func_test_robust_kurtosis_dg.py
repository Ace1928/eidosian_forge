import numpy as np
import pandas as pd
from numpy.testing import (assert_almost_equal, assert_raises, assert_equal,
from statsmodels.stats._adnorm import normal_ad
from statsmodels.stats.stattools import (omni_normtest, jarque_bera,
def test_robust_kurtosis_dg(self):
    x = self.kurtosis_x
    delta, gamma = (10.0, 45.0)
    kurtosis = robust_kurtosis(self.kurtosis_x, dg=(delta, gamma), excess=False)
    q = np.percentile(x, [delta, 100.0 - delta, gamma, 100.0 - gamma])
    assert_almost_equal(kurtosis[3], (q[1] - q[0]) / (q[3] - q[2]))