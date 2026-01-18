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
@pytest.mark.skipif(not animation.writers.is_available('ffmpeg'), reason='matplotlib animations within ArviZ require ffmpeg')
def test_plot_ppc_bad_ax(models, fig_ax):
    _, ax = fig_ax
    _, ax2 = plt.subplots(1, 2)
    with pytest.raises(ValueError, match='same figure'):
        plot_ppc(models.model_1, ax=[ax, *ax2], flatten=[], coords={'obs_dim': [1, 2, 3]}, animated=True)
    with pytest.raises(ValueError, match='2 axes'):
        plot_ppc(models.model_1, ax=ax2)