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
def test_wrap_ufunc_out_shape_multi_input():
    out_shape = (2, 4)
    func = lambda x, y: np.random.rand(*out_shape)
    ary1 = np.ones((4, 100))
    ary2 = np.ones((4, 5))
    res = wrap_xarray_ufunc(func, ary1, ary2, func_kwargs={'out_shape': out_shape}, ufunc_kwargs={'n_dims': 1})
    assert res.shape == (*ary1.shape[:-1], *out_shape)