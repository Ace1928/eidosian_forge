import os
import numpy as np
import packaging
import pandas as pd
import pytest
import scipy
from numpy.testing import assert_almost_equal
from ...data import from_cmdstan, load_arviz_data
from ...rcparams import rcParams
from ...sel_utils import xarray_var_iter
from ...stats import bfmi, ess, mcse, rhat
from ...stats.diagnostics import (
@pytest.mark.parametrize('method', ('rank', 'split', 'folded', 'z_scale', 'identity'))
def test_rhat_nan(self, method):
    """Confirm R-hat statistic returns nan."""
    data = np.random.randn(4, 100)
    data[0, 0] = np.nan
    rhat_data = rhat(data, method=method)
    assert np.isnan(rhat_data)
    if method == 'rank':
        assert np.isnan(_rhat(rhat_data))