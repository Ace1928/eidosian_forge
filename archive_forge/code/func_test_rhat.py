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
@pytest.mark.parametrize('var_names', (None, 'mu', ['mu', 'tau']))
def test_rhat(self, data, var_names, method):
    """Confirm R-hat statistic is close to 1 for a large
        number of samples. Also checks the correct shape"""
    rhat_data = rhat(data, var_names=var_names, method=method)
    for r_hat in rhat_data.data_vars.values():
        assert ((1 / GOOD_RHAT < r_hat.values) | (r_hat.values < GOOD_RHAT)).all()
    if var_names is None:
        assert list(rhat_data.data_vars) == list(data.data_vars)