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
def test_wrap_ufunc_out_shape_multi_output_diff():
    func = lambda x: (np.random.rand(5, 3), np.random.rand(10, 4))
    ary = np.ones((4, 100))
    res1, res2 = wrap_xarray_ufunc(func, ary, func_kwargs={'out_shape': ((5, 3), (10, 4))}, ufunc_kwargs={'n_dims': 1, 'n_output': 2})
    assert res1.shape == (*ary.shape[:-1], 5, 3)
    assert res2.shape == (*ary.shape[:-1], 10, 4)