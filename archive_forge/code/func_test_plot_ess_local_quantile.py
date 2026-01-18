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
@pytest.mark.parametrize('kwargs', [{'rug': True}, {'rug': True, 'rug_kind': 'max_depth', 'rug_kwargs': {'color': 'c'}}, {'extra_methods': True}, {'extra_methods': True, 'extra_kwargs': {'ls': ':'}, 'text_kwargs': {'x': 0, 'ha': 'left'}}, {'extra_methods': True, 'rug': True}])
@pytest.mark.parametrize('kind', ['local', 'quantile'])
def test_plot_ess_local_quantile(models, kind, kwargs):
    """Test specific arguments in kinds local and quantile of plot_ess."""
    idata = models.model_1
    ax = plot_ess(idata, kind=kind, backend='bokeh', show=False, **kwargs)
    assert np.all(ax)