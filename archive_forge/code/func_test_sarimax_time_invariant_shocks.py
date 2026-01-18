from statsmodels.compat.pandas import MONTH_END
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from scipy.signal import lfilter
from statsmodels.tools.sm_exceptions import (
from statsmodels.tsa.statespace import (
from .test_impulse_responses import TVSS
def test_sarimax_time_invariant_shocks(reset_randomstate):
    endog = np.arange(1, 11)
    mod = sarimax.SARIMAX(endog)
    res = mod.filter([0.5, 1.0])
    nsimulations = 10
    measurement_shocks = np.random.normal(size=nsimulations)
    state_shocks = np.random.normal(size=nsimulations)
    initial_state = res.predicted_state[:1, -1]
    actual = res.simulate(nsimulations, anchor='end', measurement_shocks=measurement_shocks, state_shocks=state_shocks, initial_state=initial_state)
    desired = lfilter([1], [1, -0.5], np.r_[initial_state, state_shocks])[:-1] + measurement_shocks
    assert_allclose(actual, desired)
    mod_actual = mod.simulate(res.params, nsimulations, anchor='end', measurement_shocks=measurement_shocks, state_shocks=state_shocks, initial_state=initial_state)
    assert_allclose(mod_actual, desired)