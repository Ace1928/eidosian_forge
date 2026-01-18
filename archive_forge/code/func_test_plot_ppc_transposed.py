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
def test_plot_ppc_transposed():
    idata = load_arviz_data('rugby')
    idata.map(lambda ds: ds.assign(points=xr.concat((ds.home_points, ds.away_points), 'field')), groups='observed_vars', inplace=True)
    assert idata.posterior_predictive.points.dims == ('field', 'chain', 'draw', 'match')
    ax = plot_ppc(idata, kind='scatter', var_names='points', flatten=['field'], coords={'match': ['Wales Italy']}, random_seed=3, num_pp_samples=8)
    x, y = ax.get_lines()[2].get_data()
    assert not np.isclose(y[0], 0)
    assert np.all(np.array([47, 44, 15, 11]) == x)