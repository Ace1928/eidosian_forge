from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
@pytest.mark.parametrize(('index', 'window'), [(period_range(start='2020-01-01 08:00', end='2020-01-01 08:08', freq='min'), '2min'), (period_range(start='2020-01-01 08:00', end='2020-01-01 12:00', freq='30min'), '1h')])
@pytest.mark.parametrize(('func', 'values'), [('min', [np.nan, 0, 0, 1, 2, 3, 4, 5, 6]), ('max', [np.nan, 0, 1, 2, 3, 4, 5, 6, 7]), ('sum', [np.nan, 0, 1, 3, 5, 7, 9, 11, 13])])
def test_rolling_period_index(index, window, func, values):
    ds = Series([0, 1, 2, 3, 4, 5, 6, 7, 8], index=index)
    result = getattr(ds.rolling(window, closed='left'), func)()
    expected = Series(values, index=index)
    tm.assert_series_equal(result, expected)