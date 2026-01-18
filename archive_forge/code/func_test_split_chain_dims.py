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
@pytest.mark.parametrize('chains', (None, 1, 2, 3))
@pytest.mark.parametrize('draws', (2, 3, 100, 101))
def test_split_chain_dims(self, chains, draws):
    if chains is None:
        data = np.random.randn(draws)
    else:
        data = np.random.randn(chains, draws)
    split_data = _split_chains(data)
    if chains is None:
        chains = 1
    assert split_data.shape == (chains * 2, draws // 2)