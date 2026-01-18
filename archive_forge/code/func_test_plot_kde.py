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
@pytest.mark.parametrize('kwargs', [{'plot_kwargs': {'line_dash': 'solid'}}, {'contour': True, 'fill_last': False}, {'contour': True, 'contourf_kwargs': {'cmap': 'plasma'}, 'contour_kwargs': {'line_width': 1}}, {'contour': False}, {'contour': False, 'pcolormesh_kwargs': {'cmap': 'plasma'}}])
def test_plot_kde(continuous_model, kwargs):
    axes = plot_kde(continuous_model['x'], continuous_model['y'], backend='bokeh', show=False, **kwargs)
    assert axes