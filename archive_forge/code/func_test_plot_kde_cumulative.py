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
@pytest.mark.parametrize('kwargs', [{'cumulative': True}, {'cumulative': True, 'plot_kwargs': {'line_dash': 'dashed'}}, {'rug': True}, {'rug': True, 'rug_kwargs': {'line_alpha': 0.2}, 'rotated': True}])
def test_plot_kde_cumulative(continuous_model, kwargs):
    axes = plot_kde(continuous_model['x'], backend='bokeh', show=False, **kwargs)
    assert axes