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
def test_plot_ess_no_sample_stats(models):
    """Test error when rug=True but sample_stats group is not present."""
    idata = models.model_1
    with pytest.raises(ValueError, match='must contain sample_stats'):
        plot_ess(idata.posterior, rug=True, backend='bokeh', show=False)