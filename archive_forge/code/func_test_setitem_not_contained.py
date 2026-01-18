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
def test_setitem_not_contained(self, string_series):
    ser = string_series.copy()
    assert 'foobar' not in ser.index
    ser['foobar'] = 1
    app = Series([1], index=['foobar'], name='series')
    expected = concat([string_series, app])
    tm.assert_series_equal(ser, expected)