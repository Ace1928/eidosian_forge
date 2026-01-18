import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_series_getitem_multiindex_xs(self):
    dt = list(date_range('20130903', periods=3))
    idx = MultiIndex.from_product([list('AB'), dt])
    ser = Series([1, 3, 4, 1, 3, 4], index=idx)
    expected = Series([1, 1], index=list('AB'))
    result = ser.xs('20130903', level=1)
    tm.assert_series_equal(result, expected)