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
@pytest.mark.parametrize('var_names', [None, 'mu', ['mu', 'tau']])
def test_plot_parallel_exception(models, var_names):
    """Ensure that correct exception is raised when one variable is passed."""
    with pytest.raises(ValueError):
        assert plot_parallel(models.model_1, var_names=var_names, norm_method='foo', backend='bokeh', show=False)