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
@pytest.mark.parametrize('kind', ['kde', 'cumulative', 'scatter'])
def test_plot_ppc_discrete_save_animation(kind):
    data = from_dict(observed_data={'obs': np.random.randint(1, 100, 15)}, posterior_predictive={'obs': np.random.randint(1, 300, (1, 20, 15))})
    animation_kwargs = {'blit': False}
    axes, anim = plot_ppc(data, kind=kind, animated=True, animation_kwargs=animation_kwargs, num_pp_samples=5, random_seed=3)
    assert axes
    assert anim
    animations_folder = '../saved_animations'
    os.makedirs(animations_folder, exist_ok=True)
    path = os.path.join(animations_folder, f'ppc_discrete_{kind}_animation.mp4')
    anim.save(path)
    assert os.path.exists(path)
    assert os.path.getsize(path)