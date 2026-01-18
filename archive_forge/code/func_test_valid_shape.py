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
def test_valid_shape():
    assert not not_valid(np.ones((2, 200)), check_nan=False, shape_kwargs=dict(min_chains=2, min_draws=100))
    assert not not_valid(np.ones((200, 2)), check_nan=False, shape_kwargs=dict(min_chains=100, min_draws=2))
    assert not_valid(np.ones((10, 10)), check_nan=False, shape_kwargs=dict(min_chains=2, min_draws=100))
    assert not_valid(np.ones((10, 10)), check_nan=False, shape_kwargs=dict(min_chains=100, min_draws=2))