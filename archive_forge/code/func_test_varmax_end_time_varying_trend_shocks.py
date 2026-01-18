from statsmodels.compat.pandas import MONTH_END
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from scipy.signal import lfilter
from statsmodels.tools.sm_exceptions import (
from statsmodels.tsa.statespace import (
from .test_impulse_responses import TVSS
def test_varmax_end_time_varying_trend_shocks(reset_randomstate):
    endog = np.arange(1, 21).reshape(10, 2)
    mod = varmax.VARMAX(endog, trend='ct')
    res = mod.filter([1.0, 1.0, 1.0, 1.0, 1, 1, 1.0, 1.0, 1.0, 0.5, 1.0])
    nsimulations = 10
    measurement_shocks = np.random.normal(size=(nsimulations, mod.k_endog))
    state_shocks = np.random.normal(size=(nsimulations, mod.k_states))
    with res._set_final_predicted_state(exog=None, out_of_sample=10):
        initial_state = res.predicted_state[:, -1].copy()
    actual = res.simulate(nsimulations, anchor='end', measurement_shocks=measurement_shocks, state_shocks=state_shocks, initial_state=initial_state)
    desired = np.zeros((nsimulations, mod.k_endog))
    desired[0] = initial_state
    tmp_trend = 1 + np.arange(11, 21)
    for i in range(1, nsimulations):
        desired[i] = desired[i - 1].sum() + tmp_trend[i] + state_shocks[i - 1]
    desired = desired + measurement_shocks
    assert_allclose(actual, desired)
    mod_actual = mod.simulate(res.params, nsimulations, anchor='end', measurement_shocks=measurement_shocks, state_shocks=state_shocks, initial_state=initial_state)
    assert_allclose(mod_actual, desired)