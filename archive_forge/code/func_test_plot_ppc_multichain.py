from copy import deepcopy
import numpy as np
import pytest
from pandas import DataFrame  # pylint: disable=wrong-import-position
from scipy.stats import norm  # pylint: disable=wrong-import-position
from ...data import from_dict, load_arviz_data  # pylint: disable=wrong-import-position
from ...plots import (  # pylint: disable=wrong-import-position
from ...rcparams import rc_context, rcParams  # pylint: disable=wrong-import-position
from ...stats import compare, hdi, loo, waic  # pylint: disable=wrong-import-position
from ..helpers import (  # pylint: disable=unused-import, wrong-import-position
@pytest.mark.parametrize('kind', ['kde', 'cumulative', 'scatter'])
@pytest.mark.parametrize('jitter', [None, 0, 0.1, 1, 3])
def test_plot_ppc_multichain(kind, jitter):
    data = from_dict(posterior_predictive={'x': np.random.randn(4, 100, 30), 'y_hat': np.random.randn(4, 100, 3, 10)}, observed_data={'x': np.random.randn(30), 'y': np.random.randn(3, 10)})
    axes = plot_ppc(data, kind=kind, data_pairs={'y': 'y_hat'}, jitter=jitter, random_seed=3, backend='bokeh', show=False)
    assert np.all(axes)