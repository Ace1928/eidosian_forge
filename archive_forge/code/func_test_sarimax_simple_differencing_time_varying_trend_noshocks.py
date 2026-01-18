from statsmodels.compat.pandas import MONTH_END
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from scipy.signal import lfilter
from statsmodels.tools.sm_exceptions import (
from statsmodels.tsa.statespace import (
from .test_impulse_responses import TVSS
def test_sarimax_simple_differencing_time_varying_trend_noshocks():
    endog = np.cumsum(np.arange(0, 11))
    mod = sarimax.SARIMAX(endog, order=(1, 1, 0), trend='t', simple_differencing=True)
    res = mod.filter([1.0, 0.2, 1.0])
    nsimulations = 10
    measurement_shocks = np.zeros((nsimulations, mod.k_endog))
    state_shocks = np.zeros((nsimulations, mod.k_states))
    initial_state = res.predicted_state[..., -1]
    assert_allclose(initial_state, 12)
    actual = res.simulate(nsimulations, anchor='end', measurement_shocks=measurement_shocks, state_shocks=state_shocks, initial_state=initial_state)
    desired = lfilter([1], [1, -0.2], np.r_[12, np.arange(11, 20)])
    assert_allclose(actual, desired)
    mod_actual = mod.simulate(res.params, nsimulations, anchor='end', measurement_shocks=measurement_shocks, state_shocks=state_shocks, initial_state=initial_state)
    assert_allclose(mod_actual, desired)
    assert_allclose(actual, res.forecast(nsimulations))