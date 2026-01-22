from datetime import (
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.frequencies import to_offset
class TestGetSliceBounds:

    @pytest.mark.parametrize('box', [date, datetime, Timestamp])
    @pytest.mark.parametrize('side, expected', [('left', 4), ('right', 5)])
    def test_get_slice_bounds_datetime_within(self, box, side, expected, tz_aware_fixture):
        tz = tz_aware_fixture
        index = bdate_range('2000-01-03', '2000-02-11').tz_localize(tz)
        key = box(year=2000, month=1, day=7)
        if tz is not None:
            with pytest.raises(TypeError, match='Cannot compare tz-naive'):
                index.get_slice_bound(key, side=side)
        else:
            result = index.get_slice_bound(key, side=side)
            assert result == expected

    @pytest.mark.parametrize('box', [datetime, Timestamp])
    @pytest.mark.parametrize('side', ['left', 'right'])
    @pytest.mark.parametrize('year, expected', [(1999, 0), (2020, 30)])
    def test_get_slice_bounds_datetime_outside(self, box, side, year, expected, tz_aware_fixture):
        tz = tz_aware_fixture
        index = bdate_range('2000-01-03', '2000-02-11').tz_localize(tz)
        key = box(year=year, month=1, day=7)
        if tz is not None:
            with pytest.raises(TypeError, match='Cannot compare tz-naive'):
                index.get_slice_bound(key, side=side)
        else:
            result = index.get_slice_bound(key, side=side)
            assert result == expected

    @pytest.mark.parametrize('box', [datetime, Timestamp])
    def test_slice_datetime_locs(self, box, tz_aware_fixture):
        tz = tz_aware_fixture
        index = DatetimeIndex(['2010-01-01', '2010-01-03']).tz_localize(tz)
        key = box(2010, 1, 1)
        if tz is not None:
            with pytest.raises(TypeError, match='Cannot compare tz-naive'):
                index.slice_locs(key, box(2010, 1, 2))
        else:
            result = index.slice_locs(key, box(2010, 1, 2))
            expected = (0, 1)
            assert result == expected