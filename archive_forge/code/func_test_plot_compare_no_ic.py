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
def test_plot_compare_no_ic(models):
    """Check exception is raised if model_compare doesn't contain a valid information criterion"""
    model_compare = compare({'Model 1': models.model_1, 'Model 2': models.model_2})
    model_compare = model_compare.drop('elpd_loo', axis=1)
    with pytest.raises(ValueError) as err:
        plot_compare(model_compare, backend='bokeh', show=False)
    assert 'comp_df must contain one of the following' in str(err.value)
    assert "['elpd_loo', 'elpd_waic']" in str(err.value)