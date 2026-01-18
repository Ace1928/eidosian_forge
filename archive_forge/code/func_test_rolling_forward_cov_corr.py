import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import (
from pandas.core.indexers.objects import (
from pandas.tseries.offsets import BusinessDay
@pytest.mark.parametrize('func,expected', [('cov', [2.0, 2.0, 2.0, 97.0, 2.0, -93.0, 2.0, 2.0, np.nan, np.nan]), ('corr', [1.0, 1.0, 1.0, 0.8704775290207161, 0.018229084250926637, -0.861357304646493, 1.0, 1.0, np.nan, np.nan])])
def test_rolling_forward_cov_corr(func, expected):
    values1 = np.arange(10).reshape(-1, 1)
    values2 = values1 * 2
    values1[5, 0] = 100
    values = np.concatenate([values1, values2], axis=1)
    indexer = FixedForwardWindowIndexer(window_size=3)
    rolling = DataFrame(values).rolling(window=indexer, min_periods=3)
    result = getattr(rolling, func)().loc[(slice(None), 1), 0]
    result = result.reset_index(drop=True)
    expected = Series(expected).reset_index(drop=True)
    expected.name = result.name
    tm.assert_equal(result, expected)