import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_series_getitem_multiindex_xs_by_label(self):
    idx = MultiIndex.from_tuples([('a', 'one'), ('a', 'two'), ('b', 'one'), ('b', 'two')])
    ser = Series([1, 2, 3, 4], index=idx)
    return_value = ser.index.set_names(['L1', 'L2'], inplace=True)
    assert return_value is None
    expected = Series([1, 3], index=['a', 'b'])
    return_value = expected.index.set_names(['L1'], inplace=True)
    assert return_value is None
    result = ser.xs('one', level='L2')
    tm.assert_series_equal(result, expected)