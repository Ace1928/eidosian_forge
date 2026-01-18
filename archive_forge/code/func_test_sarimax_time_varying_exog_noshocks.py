from statsmodels.compat.pandas import MONTH_END
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from scipy.signal import lfilter
from statsmodels.tools.sm_exceptions import (
from statsmodels.tsa.statespace import (
from .test_impulse_responses import TVSS
def test_sarimax_time_varying_exog_noshocks():
    endog = np.arange(1, 11)
    exog = np.arange(1, 21) ** 2
    mod = sarimax.SARIMAX(endog, exog=exog[:10])
    res = mod.filter([1.0, 0.2, 1.0])
    nsimulations = 10
    measurement_shocks = np.zeros((nsimulations, mod.k_endog))
    state_shocks = np.zeros((nsimulations, mod.k_states))
    initial_state = res.predicted_state[..., -1]
    actual = res.simulate(nsimulations, exog=exog[10:], anchor='end', measurement_shocks=measurement_shocks, state_shocks=state_shocks, initial_state=initial_state)
    desired = lfilter([1], [1, -0.2], np.r_[initial_state, [0] * 9]) + exog[10:]
    assert_allclose(actual, desired)
    mod_actual = mod.simulate(res.params, nsimulations, exog=exog[10:], anchor='end', measurement_shocks=measurement_shocks, state_shocks=state_shocks, initial_state=initial_state)
    assert_allclose(mod_actual, desired)
    assert_allclose(actual, res.forecast(nsimulations, exog=exog[10:]))