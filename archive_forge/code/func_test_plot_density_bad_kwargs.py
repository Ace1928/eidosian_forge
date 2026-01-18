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
def test_plot_density_bad_kwargs(models):
    obj = [getattr(models, model_fit) for model_fit in ['model_1', 'model_2']]
    with pytest.raises(ValueError):
        plot_density(obj, point_estimate='bad_value', backend='bokeh', show=False)
    with pytest.raises(ValueError):
        plot_density(obj, data_labels=[f'bad_value_{i}' for i in range(len(obj) + 10)], backend='bokeh', show=False)
    with pytest.raises(ValueError):
        plot_density(obj, hdi_prob=2, backend='bokeh', show=False)