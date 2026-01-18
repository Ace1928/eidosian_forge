import numpy as np
import pandas as pd
from numpy.testing import (assert_almost_equal, assert_raises, assert_equal,
from statsmodels.stats._adnorm import normal_ad
from statsmodels.stats.stattools import (omni_normtest, jarque_bera,
def test_robust_kurtosis(self):
    x = self.kurtosis_x
    assert_almost_equal(np.array(robust_kurtosis(x)), self.expected_kurtosis)