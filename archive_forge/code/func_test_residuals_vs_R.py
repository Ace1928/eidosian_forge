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
def test_residuals_vs_R(setup_model):
    model, params, results_R = setup_model
    yhat = model.smooth(params, return_raw=True)[0]
    residuals = model._residuals(yhat)
    assert_allclose(residuals, results_R['residuals'], rtol=1e-05, atol=1e-05)