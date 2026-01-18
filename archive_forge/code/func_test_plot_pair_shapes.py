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
@pytest.mark.parametrize('marginals', [True, False])
@pytest.mark.parametrize('max_subplots', [True, False])
def test_plot_pair_shapes(marginals, max_subplots):
    rng = np.random.default_rng()
    idata = from_dict({'a': rng.standard_normal((4, 500, 5))})
    if max_subplots:
        with rc_context({'plot.max_subplots': 6}):
            with pytest.warns(UserWarning, match='3x3 grid'):
                ax = plot_pair(idata, marginals=marginals)
    else:
        ax = plot_pair(idata, marginals=marginals)
    side = 3 if max_subplots else 4 + marginals
    assert ax.shape == (side, side)