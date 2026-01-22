import re
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import (
from pandas import (
import pandas._testing as tm
class TestDatetimelikeSubtype(AstypeTests):
    """Tests specific to IntervalIndex with datetime-like subtype"""
    indexes = [interval_range(Timestamp('2018-01-01'), periods=10, closed='neither'), interval_range(Timestamp('2018-01-01'), periods=10).insert(2, NaT), interval_range(Timestamp('2018-01-01', tz='US/Eastern'), periods=10), interval_range(Timedelta('0 days'), periods=10, closed='both'), interval_range(Timedelta('0 days'), periods=10).insert(2, NaT)]

    @pytest.fixture(params=indexes)
    def index(self, request):
        return request.param

    @pytest.mark.parametrize('subtype', ['int64', 'uint64'])
    def test_subtype_integer(self, index, subtype):
        dtype = IntervalDtype(subtype, 'right')
        if subtype != 'int64':
            msg = 'Cannot convert interval\\[(timedelta64|datetime64)\\[ns.*\\], .*\\] to interval\\[uint64, .*\\]'
            with pytest.raises(TypeError, match=msg):
                index.astype(dtype)
            return
        result = index.astype(dtype)
        new_left = index.left.astype(subtype)
        new_right = index.right.astype(subtype)
        expected = IntervalIndex.from_arrays(new_left, new_right, closed=index.closed)
        tm.assert_index_equal(result, expected)

    def test_subtype_float(self, index):
        dtype = IntervalDtype('float64', 'right')
        msg = 'Cannot convert .* to .*; subtypes are incompatible'
        with pytest.raises(TypeError, match=msg):
            index.astype(dtype)

    def test_subtype_datetimelike(self):
        dtype = IntervalDtype('timedelta64[ns]', 'right')
        msg = 'Cannot convert .* to .*; subtypes are incompatible'
        index = interval_range(Timestamp('2018-01-01'), periods=10)
        with pytest.raises(TypeError, match=msg):
            index.astype(dtype)
        index = interval_range(Timestamp('2018-01-01', tz='CET'), periods=10)
        with pytest.raises(TypeError, match=msg):
            index.astype(dtype)
        dtype = IntervalDtype('datetime64[ns]', 'right')
        index = interval_range(Timedelta('0 days'), periods=10)
        with pytest.raises(TypeError, match=msg):
            index.astype(dtype)