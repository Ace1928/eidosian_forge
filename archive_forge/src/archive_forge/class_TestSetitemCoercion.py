from __future__ import annotations
from datetime import (
import itertools
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas as pd
import pandas._testing as tm
class TestSetitemCoercion(CoercionBase):
    method = 'setitem'
    klasses: list[str] = []

    def test_setitem_series_no_coercion_from_values_list(self):
        ser = pd.Series(['a', 1])
        ser[:] = list(ser.values)
        expected = pd.Series(['a', 1])
        tm.assert_series_equal(ser, expected)

    def _assert_setitem_index_conversion(self, original_series, loc_key, expected_index, expected_dtype):
        """test index's coercion triggered by assign key"""
        temp = original_series.copy()
        temp[loc_key] = 5
        exp = pd.Series([1, 2, 3, 4, 5], index=expected_index)
        tm.assert_series_equal(temp, exp)
        assert temp.index.dtype == expected_dtype
        temp = original_series.copy()
        temp.loc[loc_key] = 5
        exp = pd.Series([1, 2, 3, 4, 5], index=expected_index)
        tm.assert_series_equal(temp, exp)
        assert temp.index.dtype == expected_dtype

    @pytest.mark.parametrize('val,exp_dtype', [('x', object), (5, IndexError), (1.1, object)])
    def test_setitem_index_object(self, val, exp_dtype):
        obj = pd.Series([1, 2, 3, 4], index=pd.Index(list('abcd'), dtype=object))
        assert obj.index.dtype == object
        if exp_dtype is IndexError:
            temp = obj.copy()
            warn_msg = 'Series.__setitem__ treating keys as positions is deprecated'
            msg = 'index 5 is out of bounds for axis 0 with size 4'
            with pytest.raises(exp_dtype, match=msg):
                with tm.assert_produces_warning(FutureWarning, match=warn_msg):
                    temp[5] = 5
        else:
            exp_index = pd.Index(list('abcd') + [val], dtype=object)
            self._assert_setitem_index_conversion(obj, val, exp_index, exp_dtype)

    @pytest.mark.parametrize('val,exp_dtype', [(5, np.int64), (1.1, np.float64), ('x', object)])
    def test_setitem_index_int64(self, val, exp_dtype):
        obj = pd.Series([1, 2, 3, 4])
        assert obj.index.dtype == np.int64
        exp_index = pd.Index([0, 1, 2, 3, val])
        self._assert_setitem_index_conversion(obj, val, exp_index, exp_dtype)

    @pytest.mark.parametrize('val,exp_dtype', [(5, np.float64), (5.1, np.float64), ('x', object)])
    def test_setitem_index_float64(self, val, exp_dtype, request):
        obj = pd.Series([1, 2, 3, 4], index=[1.1, 2.1, 3.1, 4.1])
        assert obj.index.dtype == np.float64
        exp_index = pd.Index([1.1, 2.1, 3.1, 4.1, val])
        self._assert_setitem_index_conversion(obj, val, exp_index, exp_dtype)

    @pytest.mark.xfail(reason='Test not implemented')
    def test_setitem_series_period(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason='Test not implemented')
    def test_setitem_index_complex128(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason='Test not implemented')
    def test_setitem_index_bool(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason='Test not implemented')
    def test_setitem_index_datetime64(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason='Test not implemented')
    def test_setitem_index_datetime64tz(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason='Test not implemented')
    def test_setitem_index_timedelta64(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason='Test not implemented')
    def test_setitem_index_period(self):
        raise NotImplementedError