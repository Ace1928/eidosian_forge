import numpy as np
from numpy.testing import assert_allclose, assert_equal
from scipy.stats._tukeylambda_stats import (tukeylambda_variance,
def test_tukeylambda_stats_invalid():
    """Test values of lambda outside the domains of the functions."""
    lam = [-1.0, -0.5]
    var = tukeylambda_variance(lam)
    assert_equal(var, np.array([np.nan, np.inf]))
    lam = [-1.0, -0.25]
    kurt = tukeylambda_kurtosis(lam)
    assert_equal(kurt, np.array([np.nan, np.inf]))