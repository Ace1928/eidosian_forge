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
@pytest.mark.parametrize('quantile', ((0.5,), (0.5, 0.1)))
@pytest.mark.parametrize('arg', (True, False))
def test_wrap_ufunc_output(quantile, arg):
    ary = np.random.randn(4, 100)
    n_output = len(quantile)
    if arg:
        res = wrap_xarray_ufunc(np.quantile, ary, ufunc_kwargs={'n_output': n_output}, func_args=(quantile,))
    elif n_output == 1:
        res = wrap_xarray_ufunc(np.quantile, ary, func_kwargs={'q': quantile})
    else:
        res = wrap_xarray_ufunc(np.quantile, ary, ufunc_kwargs={'n_output': n_output}, func_kwargs={'q': quantile})
    if n_output == 1:
        assert not isinstance(res, tuple)
    else:
        assert isinstance(res, tuple)
        assert len(res) == n_output