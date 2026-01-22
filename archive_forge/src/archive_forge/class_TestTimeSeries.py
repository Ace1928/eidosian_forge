from __future__ import annotations
from datetime import (
from functools import partial
from operator import attrgetter
import dateutil
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
class TestTimeSeries:

    def test_dti_constructor_preserve_dti_freq(self):
        rng = date_range('1/1/2000', '1/2/2000', freq='5min')
        rng2 = DatetimeIndex(rng)
        assert rng.freq == rng2.freq

    def test_explicit_none_freq(self):
        rng = date_range('1/1/2000', '1/2/2000', freq='5min')
        result = DatetimeIndex(rng, freq=None)
        assert result.freq is None
        result = DatetimeIndex(rng._data, freq=None)
        assert result.freq is None

    def test_dti_constructor_small_int(self, any_int_numpy_dtype):
        exp = DatetimeIndex(['1970-01-01 00:00:00.00000000', '1970-01-01 00:00:00.00000001', '1970-01-01 00:00:00.00000002'])
        arr = np.array([0, 10, 20], dtype=any_int_numpy_dtype)
        tm.assert_index_equal(DatetimeIndex(arr), exp)

    def test_ctor_str_intraday(self):
        rng = DatetimeIndex(['1-1-2000 00:00:01'])
        assert rng[0].second == 1

    def test_index_cast_datetime64_other_units(self):
        arr = np.arange(0, 100, 10, dtype=np.int64).view('M8[D]')
        idx = Index(arr)
        assert (idx.values == astype_overflowsafe(arr, dtype=np.dtype('M8[ns]'))).all()

    def test_constructor_int64_nocopy(self):
        arr = np.arange(1000, dtype=np.int64)
        index = DatetimeIndex(arr)
        arr[50:100] = -1
        assert (index.asi8[50:100] == -1).all()
        arr = np.arange(1000, dtype=np.int64)
        index = DatetimeIndex(arr, copy=True)
        arr[50:100] = -1
        assert (index.asi8[50:100] != -1).all()

    @pytest.mark.parametrize('freq', ['ME', 'QE', 'YE', 'D', 'B', 'bh', 'min', 's', 'ms', 'us', 'h', 'ns', 'C'])
    def test_from_freq_recreate_from_data(self, freq):
        org = date_range(start='2001/02/01 09:00', freq=freq, periods=1)
        idx = DatetimeIndex(org, freq=freq)
        tm.assert_index_equal(idx, org)
        org = date_range(start='2001/02/01 09:00', freq=freq, tz='US/Pacific', periods=1)
        idx = DatetimeIndex(org, freq=freq, tz='US/Pacific')
        tm.assert_index_equal(idx, org)

    def test_datetimeindex_constructor_misc(self):
        arr = ['1/1/2005', '1/2/2005', 'Jn 3, 2005', '2005-01-04']
        msg = "(\\(')?Unknown datetime string format(:', 'Jn 3, 2005'\\))?"
        with pytest.raises(ValueError, match=msg):
            DatetimeIndex(arr)
        arr = ['1/1/2005', '1/2/2005', '1/3/2005', '2005-01-04']
        idx1 = DatetimeIndex(arr)
        arr = [datetime(2005, 1, 1), '1/2/2005', '1/3/2005', '2005-01-04']
        idx2 = DatetimeIndex(arr)
        arr = [Timestamp(datetime(2005, 1, 1)), '1/2/2005', '1/3/2005', '2005-01-04']
        idx3 = DatetimeIndex(arr)
        arr = np.array(['1/1/2005', '1/2/2005', '1/3/2005', '2005-01-04'], dtype='O')
        idx4 = DatetimeIndex(arr)
        idx5 = DatetimeIndex(['12/05/2007', '25/01/2008'], dayfirst=True)
        idx6 = DatetimeIndex(['2007/05/12', '2008/01/25'], dayfirst=False, yearfirst=True)
        tm.assert_index_equal(idx5, idx6)
        for other in [idx2, idx3, idx4]:
            assert (idx1.values == other.values).all()

    def test_dti_constructor_object_dtype_dayfirst_yearfirst_with_tz(self):
        val = '5/10/16'
        dfirst = Timestamp(2016, 10, 5, tz='US/Pacific')
        yfirst = Timestamp(2005, 10, 16, tz='US/Pacific')
        result1 = DatetimeIndex([val], tz='US/Pacific', dayfirst=True)
        expected1 = DatetimeIndex([dfirst])
        tm.assert_index_equal(result1, expected1)
        result2 = DatetimeIndex([val], tz='US/Pacific', yearfirst=True)
        expected2 = DatetimeIndex([yfirst])
        tm.assert_index_equal(result2, expected2)