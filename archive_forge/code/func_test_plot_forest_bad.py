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
def test_plot_forest_bad(models, model_fits):
    obj = [getattr(models, model_fit) for model_fit in model_fits]
    with pytest.raises(TypeError):
        plot_forest(obj, kind='bad_kind', backend='bokeh', show=False)
    with pytest.raises(ValueError):
        plot_forest(obj, model_names=[f'model_name_{i}' for i in range(len(obj) + 10)], backend='bokeh', show=False)