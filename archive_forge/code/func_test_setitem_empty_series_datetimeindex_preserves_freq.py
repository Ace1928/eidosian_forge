from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p24
from pandas.errors import IndexingError
from pandas.core.dtypes.common import is_list_like
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
def test_setitem_empty_series_datetimeindex_preserves_freq(self):
    dti = DatetimeIndex([], freq='D', dtype='M8[ns]')
    series = Series([], index=dti, dtype=object)
    key = Timestamp('2012-01-01')
    series[key] = 47
    expected = Series(47, DatetimeIndex([key], freq='D').as_unit('ns'))
    tm.assert_series_equal(series, expected)
    assert series.index.freq == expected.index.freq