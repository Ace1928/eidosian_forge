from datetime import (
import numpy as np
import pytest
import pytz
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
def test_series_fillna_limit(self):
    index = np.arange(10)
    s = Series(np.random.default_rng(2).standard_normal(10), index=index)
    result = s[:2].reindex(index)
    result = result.fillna(method='pad', limit=5)
    expected = s[:2].reindex(index).fillna(method='pad')
    expected[-3:] = np.nan
    tm.assert_series_equal(result, expected)
    result = s[-2:].reindex(index)
    result = result.fillna(method='bfill', limit=5)
    expected = s[-2:].reindex(index).fillna(method='backfill')
    expected[:3] = np.nan
    tm.assert_series_equal(result, expected)