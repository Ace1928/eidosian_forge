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
@pytest.mark.parametrize('kind', ['kde', 'cumulative', 'scatter'])
def test_plot_ppc_bad(models, kind):
    data = from_dict(posterior={'mu': np.random.randn()})
    with pytest.raises(TypeError):
        plot_ppc(data, kind=kind, backend='bokeh', show=False)
    with pytest.raises(TypeError):
        plot_ppc(models.model_1, kind='bad_val', backend='bokeh', show=False)
    with pytest.raises(TypeError):
        plot_ppc(models.model_1, num_pp_samples='bad_val', backend='bokeh', show=False)