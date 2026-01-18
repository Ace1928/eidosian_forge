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
@pytest.mark.parametrize('trend', TRENDS)
@pytest.mark.parametrize('seasonal', SEASONALS)
def test_start_params(trend, seasonal):
    mod = ExponentialSmoothing(housing_data, trend=trend, seasonal=seasonal, initialization_method='estimated')
    res = mod.fit()
    res2 = mod.fit(method='basinhopping', minimize_kwargs={'minimizer_kwargs': {'method': 'SLSQP'}})
    assert isinstance(res.summary().as_text(), str)
    assert res2.sse < 1.01 * res.sse
    assert isinstance(res2.params, dict)