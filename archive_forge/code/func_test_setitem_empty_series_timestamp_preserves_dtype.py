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
def test_setitem_empty_series_timestamp_preserves_dtype(self):
    timestamp = Timestamp(1412526600000000000)
    series = Series([timestamp], index=['timestamp'], dtype=object)
    expected = series['timestamp']
    series = Series([], dtype=object)
    series['anything'] = 300.0
    series['timestamp'] = timestamp
    result = series['timestamp']
    assert result == expected