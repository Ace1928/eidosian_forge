from statsmodels.compat.pandas import QUARTER_END
from statsmodels.compat.platform import PLATFORM_LINUX32, PLATFORM_WIN
from itertools import product
import json
import pathlib
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
import pytest
import scipy.stats
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import statsmodels.tsa.holtwinters as holtwinters
import statsmodels.tsa.statespace.exponential_smoothing as statespace
def test_smooth_vs_R(setup_model):
    model, params, results_R = setup_model
    yhat, xhat = model.smooth(params, return_raw=True)
    yhat_R = results_R['fitted']
    xhat_R = get_states_from_R(results_R, model._k_states)
    assert_allclose(xhat, xhat_R, rtol=1e-05, atol=1e-05)
    assert_allclose(yhat, yhat_R, rtol=1e-05, atol=1e-05)