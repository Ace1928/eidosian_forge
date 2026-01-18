from statsmodels.compat.pandas import MONTH_END
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal
import pandas as pd
import pytest
from scipy.signal import lfilter
from statsmodels.tools.sm_exceptions import (
from statsmodels.tsa.statespace import (
from .test_impulse_responses import TVSS
def test_pandas_univariate_rangeindex():
    endog = pd.Series(np.zeros(2))
    mod = sarimax.SARIMAX(endog)
    res = mod.filter([0.5, 1.0])
    actual = res.simulate(2, state_shocks=np.zeros(2), initial_state=np.zeros(1))
    desired = pd.Series([0, 0])
    assert_allclose(actual, desired)
    actual = res.simulate(2, anchor=2, state_shocks=np.zeros(2), initial_state=np.zeros(1))
    ix = pd.RangeIndex(2, 4)
    desired = pd.Series([0, 0], index=ix)
    assert_allclose(actual, desired)
    assert_(actual.index.equals(desired.index))