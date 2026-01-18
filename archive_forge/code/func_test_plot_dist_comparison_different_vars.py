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
def test_plot_dist_comparison_different_vars():
    data = from_dict(posterior={'x': np.random.randn(4, 100, 30)}, prior={'x_hat': np.random.randn(4, 100, 30)})
    with pytest.raises(KeyError):
        plot_dist_comparison(data, var_names='x')
    ax = plot_dist_comparison(data, var_names=[['x_hat'], ['x']])
    assert np.all(ax)