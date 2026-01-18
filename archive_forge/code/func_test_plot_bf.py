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
def test_plot_bf():
    idata = from_dict(posterior={'a': np.random.normal(1, 0.5, 5000)}, prior={'a': np.random.normal(0, 1, 5000)})
    bf_dict0, _ = plot_bf(idata, var_name='a', ref_val=0)
    bf_dict1, _ = plot_bf(idata, prior=np.random.normal(0, 10, 5000), var_name='a', ref_val=0)
    assert bf_dict0['BF10'] > bf_dict0['BF01']
    assert bf_dict1['BF10'] < bf_dict1['BF01']