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
def test_plot_hdi_warning():
    """Check using both y and hdi_data sends a warning."""
    x_data = np.random.normal(0, 1, 100)
    y_data = np.random.normal(2 + x_data * 0.5, 0.5, (1, 200, 100))
    hdi_data = hdi(y_data)
    with pytest.warns(UserWarning, match='Both y and hdi_data'):
        ax = plot_hdi(x_data, y=y_data, hdi_data=hdi_data)
    assert ax