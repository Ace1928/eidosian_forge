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
def test_stats_variance_1d():
    """Test for stats_variance_1d."""
    data = np.random.rand(1000000)
    assert np.allclose(np.var(data), stats_variance_2d(data))
    assert np.allclose(np.var(data, ddof=1), stats_variance_2d(data, ddof=1))