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
def test_plot_autocorr_uncombined(models):
    axes = plot_autocorr(models.model_1, combined=False, backend='bokeh', show=False)
    assert axes.shape[0] == 10
    max_subplots = np.inf if rcParams['plot.max_subplots'] is None else rcParams['plot.max_subplots']
    assert len([ax for ax in axes.ravel() if ax is not None]) == min(72, max_subplots)