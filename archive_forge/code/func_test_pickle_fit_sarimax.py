import pickle
import numpy as np
import pandas as pd
from numpy.testing import assert_equal, assert_allclose
import pytest
from statsmodels.tsa.statespace import sarimax
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter
from statsmodels.tsa.statespace.representation import Representation
from statsmodels.tsa.statespace.structural import UnobservedComponents
from .results import results_kalman_filter
def test_pickle_fit_sarimax(data):
    mod = sarimax.SARIMAX(data['lgdp'], order=(1, 1, 0))
    pkl_mod = pickle.loads(pickle.dumps(mod))
    res = mod.fit(disp=-1, full_output=True, method='newton')
    pkl_res = pkl_mod.fit(disp=-1, full_output=True, method='newton')
    assert_allclose(res.llf_obs, pkl_res.llf_obs)
    assert_allclose(res.tvalues, pkl_res.tvalues)
    assert_allclose(res.smoothed_state, pkl_res.smoothed_state)
    assert_allclose(res.resid.values, pkl_res.resid.values)
    assert_allclose(res.impulse_responses(10), res.impulse_responses(10))