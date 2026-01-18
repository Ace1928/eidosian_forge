import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from scipy.special import logsumexp
from scipy.stats import circstd
from ...data import from_dict, load_arviz_data
from ...stats.density_utils import histogram
from ...stats.stats_utils import (
from ...stats.stats_utils import logsumexp as _logsumexp
from ...stats.stats_utils import make_ufunc, not_valid, stats_variance_2d, wrap_xarray_ufunc
def test_variance_bad_data():
    """Test for variance when the data range is extremely wide."""
    data = np.array([1e+20, 2e-08, 1e-17, 432000000000.0, 2500432, 2300000.0, 1.6e-06])
    assert np.allclose(stats_variance_2d(data), np.var(data))
    assert np.allclose(stats_variance_2d(data, ddof=1), np.var(data, ddof=1))
    assert not np.allclose(stats_variance_2d(data), np.var(data, ddof=1))