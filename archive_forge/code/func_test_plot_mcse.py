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
@pytest.mark.parametrize('kwargs', [{}, {'var_names': ['theta'], 'color': 'r'}, {'rug': True, 'rug_kwargs': {'color': 'r'}}, {'errorbar': True, 'rug': True, 'rug_kind': 'max_depth'}, {'errorbar': True, 'coords': {'school': slice(4)}, 'n_points': 10}, {'extra_methods': True, 'rug': True}, {'extra_methods': True, 'extra_kwargs': {'ls': ':'}, 'text_kwargs': {'x': 0, 'ha': 'left'}}])
def test_plot_mcse(models, kwargs):
    idata = models.model_1
    ax = plot_mcse(idata, backend='bokeh', show=False, **kwargs)
    assert np.all(ax)