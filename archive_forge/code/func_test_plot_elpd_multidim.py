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
@pytest.mark.parametrize('kwargs', [{}, {'ic': 'loo'}, {'xlabels': True, 'scale': 'log'}])
@pytest.mark.parametrize('add_model', [False, True])
@pytest.mark.parametrize('use_elpddata', [False, True])
def test_plot_elpd_multidim(multidim_models, add_model, use_elpddata, kwargs):
    model_dict = {'Model 1': multidim_models.model_1, 'Model 2': multidim_models.model_2}
    if add_model:
        model_dict['Model 3'] = create_multidimensional_model(seed=12)
    if use_elpddata:
        ic = kwargs.get('ic', 'waic')
        scale = kwargs.get('scale', 'deviance')
        if ic == 'waic':
            model_dict = {k: waic(v, scale=scale, pointwise=True) for k, v in model_dict.items()}
        else:
            model_dict = {k: loo(v, scale=scale, pointwise=True) for k, v in model_dict.items()}
    axes = plot_elpd(model_dict, backend='bokeh', show=False, **kwargs)
    assert np.any(axes)
    if add_model:
        assert axes.shape[0] == axes.shape[1]
        assert axes.shape[0] == len(model_dict) - 1