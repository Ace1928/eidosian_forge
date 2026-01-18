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
def test_plot_hdi_datetime_error():
    """Check x as datetime raises an error."""
    x_data = np.arange(start='2022-01-01', stop='2022-03-01', dtype=np.datetime64)
    y_data = np.random.normal(0, 5, (1, 200, x_data.shape[0]))
    hdi_data = hdi(y_data)
    with pytest.raises(TypeError, match='Cannot deal with x as type datetime.'):
        plot_hdi(x=x_data, y=y_data, hdi_data=hdi_data)