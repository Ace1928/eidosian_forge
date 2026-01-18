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
@pytest.mark.parametrize('var_names', (None, 'mu', ['mu', 'tau']))
@pytest.mark.parametrize('side', ['both', 'left', 'right'])
@pytest.mark.parametrize('rug', [True])
def test_plot_violin(models, var_names, side, rug):
    axes = plot_violin(models.model_1, var_names=var_names, side=side, rug=rug, backend='bokeh', show=False)
    assert axes.shape