from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
@pytest.mark.parametrize('quantile', [0.0, 0.1, 0.45, 0.5, 1])
@pytest.mark.parametrize('interpolation', ['linear', 'lower', 'higher', 'nearest', 'midpoint'])
@pytest.mark.parametrize('data', [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], [8.0, 1.0, 3.0, 4.0, 5.0, 2.0, 6.0, 7.0], [0.0, np.nan, 0.2, np.nan, 0.4], [np.nan, np.nan, np.nan, np.nan], [np.nan, 0.1, np.nan, 0.3, 0.4, 0.5], [0.5], [np.nan, 0.7, 0.6]])
def test_rolling_quantile_interpolation_options(quantile, interpolation, data):
    s = Series(data)
    q1 = s.quantile(quantile, interpolation)
    q2 = s.expanding(min_periods=1).quantile(quantile, interpolation).iloc[-1]
    if np.isnan(q1):
        assert np.isnan(q2)
    elif not IS64:
        assert np.allclose([q1], [q2], rtol=1e-07, atol=0)
    else:
        assert q1 == q2