import os
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import animation
from pandas import DataFrame
from scipy.stats import gaussian_kde, norm
import xarray as xr
from ...data import from_dict, load_arviz_data
from ...plots import (
from ...rcparams import rc_context, rcParams
from ...stats import compare, hdi, loo, waic
from ...stats.density_utils import kde as _kde
from ...utils import _cov
from ...plots.plot_utils import plot_point_interval
from ...plots.dotplot import wilkinson_algorithm
from ..helpers import (  # pylint: disable=unused-import
@pytest.mark.parametrize('kwargs', [{}, {'y_holdout': 'z', 'holdout_dim': 'holdout_dim1', 'x': ('x', 'x'), 'x_holdout': ('x_pred', 'x_pred')}, {'y_forecasts': 'z', 'holdout_dim': 'holdout_dim1'}])
def test_plot_ts_multidim(kwargs):
    """Test timeseries plots multidim functionality."""
    nchains = 4
    ndraws = 500
    ndim1 = 5
    ndim2 = 7
    data = {'y': np.random.normal(size=(ndim1, ndim2)), 'z': np.random.normal(size=(ndim1, ndim2))}
    posterior_predictive = {'y': np.random.randn(nchains, ndraws, ndim1, ndim2), 'z': np.random.randn(nchains, ndraws, ndim1, ndim2)}
    const_data = {'x': np.arange(1, 6), 'x_pred': np.arange(5, 10)}
    idata = from_dict(observed_data=data, posterior_predictive=posterior_predictive, constant_data=const_data, dims={'y': ['dim1', 'dim2'], 'z': ['holdout_dim1', 'holdout_dim2']}, coords={'dim1': range(ndim1), 'dim2': range(ndim2), 'holdout_dim1': range(ndim1 - 1, ndim1 + 4), 'holdout_dim2': range(ndim2 - 1, ndim2 + 6)})
    ax = plot_ts(idata=idata, y='y', plot_dim='dim1', **kwargs)
    assert np.all(ax)