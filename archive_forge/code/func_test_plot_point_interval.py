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
@pytest.mark.parametrize('kwargs', [{'point_estimate': 'mean', 'hdi_prob': 0.95, 'quartiles': False, 'linewidth': 2, 'markersize': 2, 'markercolor': 'red', 'marker': 'o', 'rotated': False, 'intervalcolor': 'green'}])
def test_plot_point_interval(continuous_model, kwargs):
    _, ax = plt.subplots()
    data = continuous_model['x']
    values = np.sort(data)
    ax = plot_point_interval(ax, values, **kwargs)
    assert ax