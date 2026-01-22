from __future__ import annotations
import re
import warnings
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
class TestTimedeltaArray(SharedTests):
    index_cls = TimedeltaIndex
    array_cls = TimedeltaArray
    scalar_type = pd.Timedelta
    example_dtype = 'm8[ns]'

    def test_from_tdi(self):
        tdi = TimedeltaIndex(['1 Day', '3 Hours'])
        arr = tdi._data
        assert list(arr) == list(tdi)
        tdi2 = pd.Index(arr)
        assert isinstance(tdi2, TimedeltaIndex)
        assert list(tdi2) == list(arr)

    def test_astype_object(self):
        tdi = TimedeltaIndex(['1 Day', '3 Hours'])
        arr = tdi._data
        asobj = arr.astype('O')
        assert isinstance(asobj, np.ndarray)
        assert asobj.dtype == 'O'
        assert list(asobj) == list(tdi)

    def test_to_pytimedelta(self, timedelta_index):
        tdi = timedelta_index
        arr = tdi._data
        expected = tdi.to_pytimedelta()
        result = arr.to_pytimedelta()
        tm.assert_numpy_array_equal(result, expected)

    def test_total_seconds(self, timedelta_index):
        tdi = timedelta_index
        arr = tdi._data
        expected = tdi.total_seconds()
        result = arr.total_seconds()
        tm.assert_numpy_array_equal(result, expected.values)

    @pytest.mark.parametrize('propname', TimedeltaArray._field_ops)
    def test_int_properties(self, timedelta_index, propname):
        tdi = timedelta_index
        arr = tdi._data
        result = getattr(arr, propname)
        expected = np.array(getattr(tdi, propname), dtype=result.dtype)
        tm.assert_numpy_array_equal(result, expected)

    def test_array_interface(self, timedelta_index):
        arr = timedelta_index._data
        result = np.asarray(arr)
        expected = arr._ndarray
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(arr, copy=False)
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)
        result = np.asarray(arr, dtype='timedelta64[ns]')
        expected = arr._ndarray
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(arr, dtype='timedelta64[ns]', copy=False)
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(arr, dtype='timedelta64[ns]')
        assert result is not expected
        tm.assert_numpy_array_equal(result, expected)
        result = np.asarray(arr, dtype=object)
        expected = np.array(list(arr), dtype=object)
        tm.assert_numpy_array_equal(result, expected)
        result = np.asarray(arr, dtype='int64')
        assert result is not arr.asi8
        assert not np.may_share_memory(arr, result)
        expected = arr.asi8.copy()
        tm.assert_numpy_array_equal(result, expected)
        for dtype in ['float64', str]:
            result = np.asarray(arr, dtype=dtype)
            expected = np.asarray(arr).astype(dtype)
            tm.assert_numpy_array_equal(result, expected)

    def test_take_fill_valid(self, timedelta_index, fixed_now_ts):
        tdi = timedelta_index
        arr = tdi._data
        td1 = pd.Timedelta(days=1)
        result = arr.take([-1, 1], allow_fill=True, fill_value=td1)
        assert result[0] == td1
        value = fixed_now_ts
        msg = f"value should be a '{arr._scalar_type.__name__}' or 'NaT'. Got"
        with pytest.raises(TypeError, match=msg):
            arr.take([0, 1], allow_fill=True, fill_value=value)
        value = fixed_now_ts.to_period('D')
        with pytest.raises(TypeError, match=msg):
            arr.take([0, 1], allow_fill=True, fill_value=value)
        value = np.datetime64('NaT', 'ns')
        with pytest.raises(TypeError, match=msg):
            arr.take([-1, 1], allow_fill=True, fill_value=value)