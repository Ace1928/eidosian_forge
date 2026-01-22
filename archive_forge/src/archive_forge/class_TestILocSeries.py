from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
class TestILocSeries:

    def test_iloc(self, using_copy_on_write, warn_copy_on_write):
        ser = Series(np.random.default_rng(2).standard_normal(10), index=list(range(0, 20, 2)))
        ser_original = ser.copy()
        for i in range(len(ser)):
            result = ser.iloc[i]
            exp = ser[ser.index[i]]
            tm.assert_almost_equal(result, exp)
        result = ser.iloc[slice(1, 3)]
        expected = ser.loc[2:4]
        tm.assert_series_equal(result, expected)
        with tm.assert_produces_warning(None):
            with tm.assert_cow_warning(warn_copy_on_write):
                result[:] = 0
        if using_copy_on_write:
            tm.assert_series_equal(ser, ser_original)
        else:
            assert (ser.iloc[1:3] == 0).all()
        result = ser.iloc[[0, 2, 3, 4, 5]]
        expected = ser.reindex(ser.index[[0, 2, 3, 4, 5]])
        tm.assert_series_equal(result, expected)

    def test_iloc_getitem_nonunique(self):
        ser = Series([0, 1, 2], index=[0, 1, 0])
        assert ser.iloc[2] == 2

    def test_iloc_setitem_pure_position_based(self):
        ser1 = Series([1, 2, 3])
        ser2 = Series([4, 5, 6], index=[1, 0, 2])
        ser1.iloc[1:3] = ser2.iloc[1:3]
        expected = Series([1, 5, 6])
        tm.assert_series_equal(ser1, expected)

    def test_iloc_nullable_int64_size_1_nan(self):
        result = DataFrame({'a': ['test'], 'b': [np.nan]})
        with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
            result.loc[:, 'b'] = result.loc[:, 'b'].astype('Int64')
        expected = DataFrame({'a': ['test'], 'b': array([NA], dtype='Int64')})
        tm.assert_frame_equal(result, expected)