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
@pytest.mark.parametrize('model_fits', [['model_1'], ['model_1', 'model_2']])
@pytest.mark.parametrize('args_expected', [({}, 1), ({'var_names': 'mu'}, 1), ({'var_names': 'mu', 'rope': (-1, 1)}, 1), ({'r_hat': True, 'quartiles': False}, 2), ({'var_names': ['mu'], 'colors': 'black', 'ess': True, 'combined': True}, 2), ({'kind': 'ridgeplot', 'ridgeplot_truncate': False, 'ridgeplot_quantiles': [0.25, 0.5, 0.75]}, 1), ({'kind': 'ridgeplot', 'r_hat': True, 'ess': True}, 3), ({'kind': 'ridgeplot', 'r_hat': True, 'ess': True, 'ridgeplot_alpha': 0}, 3), ({'var_names': ['mu', 'tau'], 'rope': {'mu': [{'rope': (-0.1, 0.1)}], 'theta': [{'school': 'Choate', 'rope': (0.2, 0.5)}]}}, 1)])
def test_plot_forest(models, model_fits, args_expected):
    obj = [getattr(models, model_fit) for model_fit in model_fits]
    args, expected = args_expected
    axes = plot_forest(obj, backend='bokeh', show=False, **args)
    assert axes.shape == (1, expected)