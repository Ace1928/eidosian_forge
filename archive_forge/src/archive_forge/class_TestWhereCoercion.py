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
class TestWhereCoercion(CoercionBase):
    method = 'where'
    _cond = np.array([True, False, True, False])

    def _assert_where_conversion(self, original, cond, values, expected, expected_dtype):
        """test coercion triggered by where"""
        target = original.copy()
        res = target.where(cond, values)
        tm.assert_equal(res, expected)
        assert res.dtype == expected_dtype

    def _construct_exp(self, obj, klass, fill_val, exp_dtype):
        if fill_val is True:
            values = klass([True, False, True, True])
        elif isinstance(fill_val, (datetime, np.datetime64)):
            values = pd.date_range(fill_val, periods=4)
        else:
            values = klass((x * fill_val for x in [5, 6, 7, 8]))
        exp = klass([obj[0], values[1], obj[2], values[3]], dtype=exp_dtype)
        return (values, exp)

    def _run_test(self, obj, fill_val, klass, exp_dtype):
        cond = klass(self._cond)
        exp = klass([obj[0], fill_val, obj[2], fill_val], dtype=exp_dtype)
        self._assert_where_conversion(obj, cond, fill_val, exp, exp_dtype)
        values, exp = self._construct_exp(obj, klass, fill_val, exp_dtype)
        self._assert_where_conversion(obj, cond, values, exp, exp_dtype)

    @pytest.mark.parametrize('fill_val,exp_dtype', [(1, object), (1.1, object), (1 + 1j, object), (True, object)])
    def test_where_object(self, index_or_series, fill_val, exp_dtype):
        klass = index_or_series
        obj = klass(list('abcd'), dtype=object)
        assert obj.dtype == object
        self._run_test(obj, fill_val, klass, exp_dtype)

    @pytest.mark.parametrize('fill_val,exp_dtype', [(1, np.int64), (1.1, np.float64), (1 + 1j, np.complex128), (True, object)])
    def test_where_int64(self, index_or_series, fill_val, exp_dtype, request):
        klass = index_or_series
        obj = klass([1, 2, 3, 4])
        assert obj.dtype == np.int64
        self._run_test(obj, fill_val, klass, exp_dtype)

    @pytest.mark.parametrize('fill_val, exp_dtype', [(1, np.float64), (1.1, np.float64), (1 + 1j, np.complex128), (True, object)])
    def test_where_float64(self, index_or_series, fill_val, exp_dtype, request):
        klass = index_or_series
        obj = klass([1.1, 2.2, 3.3, 4.4])
        assert obj.dtype == np.float64
        self._run_test(obj, fill_val, klass, exp_dtype)

    @pytest.mark.parametrize('fill_val,exp_dtype', [(1, np.complex128), (1.1, np.complex128), (1 + 1j, np.complex128), (True, object)])
    def test_where_complex128(self, index_or_series, fill_val, exp_dtype):
        klass = index_or_series
        obj = klass([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], dtype=np.complex128)
        assert obj.dtype == np.complex128
        self._run_test(obj, fill_val, klass, exp_dtype)

    @pytest.mark.parametrize('fill_val,exp_dtype', [(1, object), (1.1, object), (1 + 1j, object), (True, np.bool_)])
    def test_where_series_bool(self, index_or_series, fill_val, exp_dtype):
        klass = index_or_series
        obj = klass([True, False, True, False])
        assert obj.dtype == np.bool_
        self._run_test(obj, fill_val, klass, exp_dtype)

    @pytest.mark.parametrize('fill_val,exp_dtype', [(pd.Timestamp('2012-01-01'), 'datetime64[ns]'), (pd.Timestamp('2012-01-01', tz='US/Eastern'), object)], ids=['datetime64', 'datetime64tz'])
    def test_where_datetime64(self, index_or_series, fill_val, exp_dtype):
        klass = index_or_series
        obj = klass(pd.date_range('2011-01-01', periods=4, freq='D')._with_freq(None))
        assert obj.dtype == 'datetime64[ns]'
        fv = fill_val
        if exp_dtype == 'datetime64[ns]':
            for scalar in [fv, fv.to_pydatetime(), fv.to_datetime64()]:
                self._run_test(obj, scalar, klass, exp_dtype)
        else:
            for scalar in [fv, fv.to_pydatetime()]:
                self._run_test(obj, fill_val, klass, exp_dtype)

    @pytest.mark.xfail(reason='Test not implemented')
    def test_where_index_complex128(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason='Test not implemented')
    def test_where_index_bool(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason='Test not implemented')
    def test_where_series_timedelta64(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason='Test not implemented')
    def test_where_series_period(self):
        raise NotImplementedError

    @pytest.mark.parametrize('value', [pd.Timedelta(days=9), timedelta(days=9), np.timedelta64(9, 'D')])
    def test_where_index_timedelta64(self, value):
        tdi = pd.timedelta_range('1 Day', periods=4)
        cond = np.array([True, False, False, True])
        expected = pd.TimedeltaIndex(['1 Day', value, value, '4 Days'])
        result = tdi.where(cond, value)
        tm.assert_index_equal(result, expected)
        dtnat = np.datetime64('NaT', 'ns')
        expected = pd.Index([tdi[0], dtnat, dtnat, tdi[3]], dtype=object)
        assert expected[1] is dtnat
        result = tdi.where(cond, dtnat)
        tm.assert_index_equal(result, expected)

    def test_where_index_period(self):
        dti = pd.date_range('2016-01-01', periods=3, freq='QS')
        pi = dti.to_period('Q')
        cond = np.array([False, True, False])
        value = pi[-1] + pi.freq * 10
        expected = pd.PeriodIndex([value, pi[1], value])
        result = pi.where(cond, value)
        tm.assert_index_equal(result, expected)
        other = np.asarray(pi + pi.freq * 10, dtype=object)
        result = pi.where(cond, other)
        expected = pd.PeriodIndex([other[0], pi[1], other[2]])
        tm.assert_index_equal(result, expected)
        td = pd.Timedelta(days=4)
        expected = pd.Index([td, pi[1], td], dtype=object)
        result = pi.where(cond, td)
        tm.assert_index_equal(result, expected)
        per = pd.Period('2020-04-21', 'D')
        expected = pd.Index([per, pi[1], per], dtype=object)
        result = pi.where(cond, per)
        tm.assert_index_equal(result, expected)