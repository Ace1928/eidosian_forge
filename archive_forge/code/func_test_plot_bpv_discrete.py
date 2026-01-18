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
def test_plot_bpv_discrete():
    fake_obs = {'a': np.random.poisson(2.5, 100)}
    fake_pp = {'a': np.random.poisson(2.5, (1, 10, 100))}
    fake_model = from_dict(posterior_predictive=fake_pp, observed_data=fake_obs)
    axes = plot_bpv(fake_model, backend='bokeh', show=False)
    assert axes.shape