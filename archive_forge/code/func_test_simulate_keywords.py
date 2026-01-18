from statsmodels.compat.pandas import MONTH_END
from statsmodels.compat.pytest import pytest_warns
import os
import re
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
import pytest
import scipy.stats
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from statsmodels.tsa.holtwinters import (
from statsmodels.tsa.holtwinters._exponential_smoothers import (
from statsmodels.tsa.holtwinters._smoothers import (
def test_simulate_keywords(austourists):
    """
    check whether all keywords are accepted and work without throwing errors.
    """
    fit = ExponentialSmoothing(austourists, seasonal_periods=4, trend='add', seasonal='add', damped_trend=True, initialization_method='estimated').fit()
    assert_almost_equal(fit.simulate(4, anchor=0, random_state=0).values, fit.simulate(4, anchor='start', random_state=0).values)
    assert_almost_equal(fit.simulate(4, anchor=-1, random_state=0).values, fit.simulate(4, anchor='2015-12-01', random_state=0).values)
    assert_almost_equal(fit.simulate(4, anchor='end', random_state=0).values, fit.simulate(4, anchor='2016-03-01', random_state=0).values)
    fit.simulate(4, repetitions=10, random_errors=scipy.stats.norm)
    fit.simulate(4, repetitions=10, random_errors=scipy.stats.norm())
    fit.simulate(4, repetitions=10, random_errors=np.random.randn(4, 10))
    fit.simulate(4, repetitions=10, random_errors='bootstrap')
    res = fit.simulate(4, repetitions=10, random_state=10).values
    res2 = fit.simulate(4, repetitions=10, random_state=np.random.RandomState(10)).values
    assert np.all(res == res2)