import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_to_xarray_with_multiindex(self):
    from xarray import DataArray
    mi = MultiIndex.from_product([['a', 'b'], range(3)], names=['one', 'two'])
    ser = Series(range(6), dtype='int64', index=mi)
    result = ser.to_xarray()
    assert len(result) == 2
    tm.assert_almost_equal(list(result.coords.keys()), ['one', 'two'])
    assert isinstance(result, DataArray)
    res = result.to_series()
    tm.assert_series_equal(res, ser)