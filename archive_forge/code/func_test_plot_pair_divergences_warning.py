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
@pytest.mark.parametrize('has_sample_stats', [True, False])
def test_plot_pair_divergences_warning(has_sample_stats):
    data = load_arviz_data('centered_eight')
    if has_sample_stats:
        data.sample_stats = data.sample_stats.rename({'diverging': 'diverging_missing'})
    else:
        data = data.posterior
    with pytest.warns(UserWarning):
        ax = plot_pair(data, divergences=True, backend='bokeh', show=False)
    assert np.any(ax)