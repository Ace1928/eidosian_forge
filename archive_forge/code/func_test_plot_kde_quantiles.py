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
@pytest.mark.parametrize('kwargs', [{'plot_kwargs': {'line_dash': 'solid'}}, {'cumulative': True}, {'rug': True}])
def test_plot_kde_quantiles(continuous_model, kwargs):
    axes = plot_kde(continuous_model['x'], quantiles=[0.05, 0.5, 0.95], backend='bokeh', show=False, **kwargs)
    assert axes