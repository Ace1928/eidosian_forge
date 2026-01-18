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
@pytest.mark.parametrize('val_err_kwargs', [{}, {'kind_pp': 'bad_kind'}, {'kind_model': 'bad_kind'}])
def test_plot_lm_valueerror(multidim_models, val_err_kwargs):
    """Test error plot_dim gets no value for multidim data and wrong value in kind_... args."""
    idata2 = multidim_models.model_1
    with pytest.raises(ValueError):
        plot_lm(idata=idata2, y='y', **val_err_kwargs)