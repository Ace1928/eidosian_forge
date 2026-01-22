from contextlib import nullcontext
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
from pandas._config import config as cf
from pandas._libs import missing as libmissing
from pandas._libs.tslibs import iNaT
from pandas.compat.numpy import np_version_gte1p25
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import (
import pandas as pd
from pandas import (
import pandas._testing as tm
class TestIsNA:

    def test_0d_array(self):
        assert isna(np.array(np.nan))
        assert not isna(np.array(0.0))
        assert not isna(np.array(0))
        assert isna(np.array(np.nan, dtype=object))
        assert not isna(np.array(0.0, dtype=object))
        assert not isna(np.array(0, dtype=object))

    @pytest.mark.parametrize('shape', [(4, 0), (4,)])
    def test_empty_object(self, shape):
        arr = np.empty(shape=shape, dtype=object)
        result = isna(arr)
        expected = np.ones(shape=shape, dtype=bool)
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('isna_f', [isna, isnull])
    def test_isna_isnull(self, isna_f):
        assert not isna_f(1.0)
        assert isna_f(None)
        assert isna_f(np.nan)
        assert float('nan')
        assert not isna_f(np.inf)
        assert not isna_f(-np.inf)
        assert not isna_f(type(Series(dtype=object)))
        assert not isna_f(type(Series(dtype=np.float64)))
        assert not isna_f(type(pd.DataFrame()))

    @pytest.mark.parametrize('isna_f', [isna, isnull])
    @pytest.mark.parametrize('data', [np.arange(4, dtype=float), [0.0, 1.0, 0.0, 1.0], Series(list('abcd'), dtype=object), date_range('2020-01-01', periods=4)])
    @pytest.mark.parametrize('index', [date_range('2020-01-01', periods=4), range(4), period_range('2020-01-01', periods=4)])
    def test_isna_isnull_frame(self, isna_f, data, index):
        df = pd.DataFrame(data, index=index)
        result = isna_f(df)
        expected = df.apply(isna_f)
        tm.assert_frame_equal(result, expected)

    def test_isna_lists(self):
        result = isna([[False]])
        exp = np.array([[False]])
        tm.assert_numpy_array_equal(result, exp)
        result = isna([[1], [2]])
        exp = np.array([[False], [False]])
        tm.assert_numpy_array_equal(result, exp)
        result = isna(['foo', 'bar'])
        exp = np.array([False, False])
        tm.assert_numpy_array_equal(result, exp)
        result = isna(['foo', 'bar'])
        exp = np.array([False, False])
        tm.assert_numpy_array_equal(result, exp)
        result = isna([np.nan, 'world'])
        exp = np.array([True, False])
        tm.assert_numpy_array_equal(result, exp)

    def test_isna_nat(self):
        result = isna([NaT])
        exp = np.array([True])
        tm.assert_numpy_array_equal(result, exp)
        result = isna(np.array([NaT], dtype=object))
        exp = np.array([True])
        tm.assert_numpy_array_equal(result, exp)

    def test_isna_numpy_nat(self):
        arr = np.array([NaT, np.datetime64('NaT'), np.timedelta64('NaT'), np.datetime64('NaT', 's')])
        result = isna(arr)
        expected = np.array([True] * 4)
        tm.assert_numpy_array_equal(result, expected)

    def test_isna_datetime(self):
        assert not isna(datetime.now())
        assert notna(datetime.now())
        idx = date_range('1/1/1990', periods=20)
        exp = np.ones(len(idx), dtype=bool)
        tm.assert_numpy_array_equal(notna(idx), exp)
        idx = np.asarray(idx)
        idx[0] = iNaT
        idx = DatetimeIndex(idx)
        mask = isna(idx)
        assert mask[0]
        exp = np.array([True] + [False] * (len(idx) - 1), dtype=bool)
        tm.assert_numpy_array_equal(mask, exp)
        pidx = idx.to_period(freq='M')
        mask = isna(pidx)
        assert mask[0]
        exp = np.array([True] + [False] * (len(idx) - 1), dtype=bool)
        tm.assert_numpy_array_equal(mask, exp)
        mask = isna(pidx[1:])
        exp = np.zeros(len(mask), dtype=bool)
        tm.assert_numpy_array_equal(mask, exp)

    def test_isna_old_datetimelike(self):
        dti = date_range('2016-01-01', periods=3)
        dta = dti._data
        dta[-1] = NaT
        expected = np.array([False, False, True], dtype=bool)
        objs = [dta, dta.tz_localize('US/Eastern'), dta - dta, dta.to_period('D')]
        for obj in objs:
            msg = 'use_inf_as_na option is deprecated'
            with tm.assert_produces_warning(FutureWarning, match=msg):
                with cf.option_context('mode.use_inf_as_na', True):
                    result = isna(obj)
            tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize('value, expected', [(np.complex128(np.nan), True), (np.float64(1), False), (np.array([1, 1 + 0j, np.nan, 3]), np.array([False, False, True, False])), (np.array([1, 1 + 0j, np.nan, 3], dtype=object), np.array([False, False, True, False])), (np.array([1, 1 + 0j, np.nan, 3]).astype(object), np.array([False, False, True, False]))])
    def test_complex(self, value, expected):
        result = isna(value)
        if is_scalar(result):
            assert result is expected
        else:
            tm.assert_numpy_array_equal(result, expected)

    def test_datetime_other_units(self):
        idx = DatetimeIndex(['2011-01-01', 'NaT', '2011-01-02'])
        exp = np.array([False, True, False])
        tm.assert_numpy_array_equal(isna(idx), exp)
        tm.assert_numpy_array_equal(notna(idx), ~exp)
        tm.assert_numpy_array_equal(isna(idx.values), exp)
        tm.assert_numpy_array_equal(notna(idx.values), ~exp)

    @pytest.mark.parametrize('dtype', ['datetime64[D]', 'datetime64[h]', 'datetime64[m]', 'datetime64[s]', 'datetime64[ms]', 'datetime64[us]', 'datetime64[ns]'])
    def test_datetime_other_units_astype(self, dtype):
        idx = DatetimeIndex(['2011-01-01', 'NaT', '2011-01-02'])
        values = idx.values.astype(dtype)
        exp = np.array([False, True, False])
        tm.assert_numpy_array_equal(isna(values), exp)
        tm.assert_numpy_array_equal(notna(values), ~exp)
        exp = Series([False, True, False])
        s = Series(values)
        tm.assert_series_equal(isna(s), exp)
        tm.assert_series_equal(notna(s), ~exp)
        s = Series(values, dtype=object)
        tm.assert_series_equal(isna(s), exp)
        tm.assert_series_equal(notna(s), ~exp)

    def test_timedelta_other_units(self):
        idx = TimedeltaIndex(['1 days', 'NaT', '2 days'])
        exp = np.array([False, True, False])
        tm.assert_numpy_array_equal(isna(idx), exp)
        tm.assert_numpy_array_equal(notna(idx), ~exp)
        tm.assert_numpy_array_equal(isna(idx.values), exp)
        tm.assert_numpy_array_equal(notna(idx.values), ~exp)

    @pytest.mark.parametrize('dtype', ['timedelta64[D]', 'timedelta64[h]', 'timedelta64[m]', 'timedelta64[s]', 'timedelta64[ms]', 'timedelta64[us]', 'timedelta64[ns]'])
    def test_timedelta_other_units_dtype(self, dtype):
        idx = TimedeltaIndex(['1 days', 'NaT', '2 days'])
        values = idx.values.astype(dtype)
        exp = np.array([False, True, False])
        tm.assert_numpy_array_equal(isna(values), exp)
        tm.assert_numpy_array_equal(notna(values), ~exp)
        exp = Series([False, True, False])
        s = Series(values)
        tm.assert_series_equal(isna(s), exp)
        tm.assert_series_equal(notna(s), ~exp)
        s = Series(values, dtype=object)
        tm.assert_series_equal(isna(s), exp)
        tm.assert_series_equal(notna(s), ~exp)

    def test_period(self):
        idx = pd.PeriodIndex(['2011-01', 'NaT', '2012-01'], freq='M')
        exp = np.array([False, True, False])
        tm.assert_numpy_array_equal(isna(idx), exp)
        tm.assert_numpy_array_equal(notna(idx), ~exp)
        exp = Series([False, True, False])
        s = Series(idx)
        tm.assert_series_equal(isna(s), exp)
        tm.assert_series_equal(notna(s), ~exp)
        s = Series(idx, dtype=object)
        tm.assert_series_equal(isna(s), exp)
        tm.assert_series_equal(notna(s), ~exp)

    def test_decimal(self):
        a = Decimal(1.0)
        assert isna(a) is False
        assert notna(a) is True
        b = Decimal('NaN')
        assert isna(b) is True
        assert notna(b) is False
        arr = np.array([a, b])
        expected = np.array([False, True])
        result = isna(arr)
        tm.assert_numpy_array_equal(result, expected)
        result = notna(arr)
        tm.assert_numpy_array_equal(result, ~expected)
        ser = Series(arr)
        expected = Series(expected)
        result = isna(ser)
        tm.assert_series_equal(result, expected)
        result = notna(ser)
        tm.assert_series_equal(result, ~expected)
        idx = Index(arr)
        expected = np.array([False, True])
        result = isna(idx)
        tm.assert_numpy_array_equal(result, expected)
        result = notna(idx)
        tm.assert_numpy_array_equal(result, ~expected)