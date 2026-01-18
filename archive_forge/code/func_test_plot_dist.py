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
@pytest.mark.parametrize('kwargs', [{'kind': 'hist'}, {'kind': 'kde'}, {'is_circular': False}, {'is_circular': False, 'kind': 'hist'}, {'is_circular': True}, {'is_circular': True, 'kind': 'hist'}, {'is_circular': 'radians'}, {'is_circular': 'radians', 'kind': 'hist'}, {'is_circular': 'degrees'}, {'is_circular': 'degrees', 'kind': 'hist'}])
def test_plot_dist(continuous_model, kwargs):
    axes = plot_dist(continuous_model['x'], backend='bokeh', show=False, **kwargs)
    assert axes