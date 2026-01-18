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
def test_plot_density_nonstring_varnames():
    """Test plot_density works when variables are not strings."""
    rv1 = RandomVariableTestClass('a')
    rv2 = RandomVariableTestClass('b')
    rv3 = RandomVariableTestClass('c')
    model_ab = from_dict({rv1: np.random.normal(size=200), rv2: np.random.normal(size=200)})
    model_bc = from_dict({rv2: np.random.normal(size=200), rv3: np.random.normal(size=200)})
    axes = plot_density([model_ab, model_bc])
    assert axes.size == 3