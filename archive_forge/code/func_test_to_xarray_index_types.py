import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_to_xarray_index_types(self, index_flat):
    index = index_flat
    from xarray import DataArray
    ser = Series(range(len(index)), index=index, dtype='int64')
    ser.index.name = 'foo'
    result = ser.to_xarray()
    repr(result)
    assert len(result) == len(index)
    assert len(result.coords) == 1
    tm.assert_almost_equal(list(result.coords.keys()), ['foo'])
    assert isinstance(result, DataArray)
    tm.assert_series_equal(result.to_series(), ser)