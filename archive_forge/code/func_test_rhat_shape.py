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
@pytest.mark.parametrize('chain', (None, 1, 2))
@pytest.mark.parametrize('draw', (1, 2, 3, 4))
def test_rhat_shape(self, method, chain, draw):
    """Confirm R-hat statistic returns nan."""
    data = np.random.randn(draw) if chain is None else np.random.randn(chain, draw)
    if chain in (None, 1) or draw < 4:
        rhat_data = rhat(data, method=method)
        assert np.isnan(rhat_data)
    else:
        rhat_data = rhat(data, method=method)
        assert not np.isnan(rhat_data)