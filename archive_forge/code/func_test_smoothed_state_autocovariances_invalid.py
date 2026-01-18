import os
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
import pandas as pd
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import mlemodel, sarimax, varmax
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
from statsmodels.tsa.statespace.kalman_filter import FILTER_UNIVARIATE
from statsmodels.tsa.statespace.kalman_smoother import (
def test_smoothed_state_autocovariances_invalid():
    _, res = get_acov_model(missing=False, filter_univariate=False, tvp=False)
    with pytest.raises(ValueError, match='Cannot specify both `t`'):
        res.smoothed_state_autocovariance(1, t=1, start=1)
    with pytest.raises(ValueError, match='Negative `t`'):
        res.smoothed_state_autocovariance(1, t=-1)
    with pytest.raises(ValueError, match='Negative `t`'):
        res.smoothed_state_autocovariance(1, start=-1)
    with pytest.raises(ValueError, match='Negative `t`'):
        res.smoothed_state_autocovariance(1, end=-1)
    with pytest.raises(ValueError, match='`end` must be after `start`'):
        res.smoothed_state_autocovariance(1, start=5, end=4)