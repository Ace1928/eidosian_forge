import collections
from datetime import timedelta
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.base.common import allow_na_ops
def test_value_counts_timedelta64(index_or_series, unit):
    klass = index_or_series
    day = Timedelta(timedelta(1)).as_unit(unit)
    tdi = TimedeltaIndex([day], name='dt').as_unit(unit)
    tdvals = np.zeros(6, dtype=f'm8[{unit}]') + day
    td = klass(tdvals, name='dt')
    result = td.value_counts()
    expected_s = Series([6], index=tdi, name='count')
    tm.assert_series_equal(result, expected_s)
    expected = tdi
    result = td.unique()
    if isinstance(td, Index):
        tm.assert_index_equal(result, expected)
    else:
        tm.assert_extension_array_equal(result, expected._values)
    td2 = day + np.zeros(6, dtype=f'm8[{unit}]')
    td2 = klass(td2, name='dt')
    result2 = td2.value_counts()
    tm.assert_series_equal(result2, expected_s)