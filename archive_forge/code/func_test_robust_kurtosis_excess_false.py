import numpy as np
import pandas as pd
from numpy.testing import (assert_almost_equal, assert_raises, assert_equal,
from statsmodels.stats._adnorm import normal_ad
from statsmodels.stats.stattools import (omni_normtest, jarque_bera,
def test_robust_kurtosis_excess_false(self):
    x = self.kurtosis_x
    expected = self.expected_kurtosis + self.kurtosis_constants
    kurtosis = np.array(robust_kurtosis(x, excess=False))
    assert_almost_equal(expected, kurtosis)