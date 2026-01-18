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
def test_representation_pickle():
    nobs = 10
    k_endog = 2
    arr = np.arange(nobs * k_endog).reshape(k_endog, nobs) * 1.0
    endog = np.asfortranarray(arr)
    mod = Representation(endog, k_states=2)
    pkl_mod = pickle.loads(pickle.dumps(mod))
    assert_equal(mod.nobs, pkl_mod.nobs)
    assert_equal(mod.k_endog, pkl_mod.k_endog)
    mod._initialize_representation()
    pkl_mod._initialize_representation()
    assert_equal(mod.design, pkl_mod.design)
    assert_equal(mod.obs_intercept, pkl_mod.obs_intercept)
    assert_equal(mod.initial_variance, pkl_mod.initial_variance)