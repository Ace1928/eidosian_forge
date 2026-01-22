from datetime import (
from itertools import (
import operator
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.conversion import localize_pydatetime
from pandas._libs.tslibs.offsets import shift_months
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import roperator
from pandas.tests.arithmetic.common import (
class TestDatetime64OverflowHandling:

    def test_dt64_overflow_masking(self, box_with_array):
        left = Series([Timestamp('1969-12-31')], dtype='M8[ns]')
        right = Series([NaT])
        left = tm.box_expected(left, box_with_array)
        right = tm.box_expected(right, box_with_array)
        expected = TimedeltaIndex([NaT], dtype='m8[ns]')
        expected = tm.box_expected(expected, box_with_array)
        result = left - right
        tm.assert_equal(result, expected)

    def test_dt64_series_arith_overflow(self):
        dt = Timestamp('1700-01-31')
        td = Timedelta('20000 Days')
        dti = date_range('1949-09-30', freq='100YE', periods=4)
        ser = Series(dti)
        msg = 'Overflow in int64 addition'
        with pytest.raises(OverflowError, match=msg):
            ser - dt
        with pytest.raises(OverflowError, match=msg):
            dt - ser
        with pytest.raises(OverflowError, match=msg):
            ser + td
        with pytest.raises(OverflowError, match=msg):
            td + ser
        ser.iloc[-1] = NaT
        expected = Series(['2004-10-03', '2104-10-04', '2204-10-04', 'NaT'], dtype='datetime64[ns]')
        res = ser + td
        tm.assert_series_equal(res, expected)
        res = td + ser
        tm.assert_series_equal(res, expected)
        ser.iloc[1:] = NaT
        expected = Series(['91279 Days', 'NaT', 'NaT', 'NaT'], dtype='timedelta64[ns]')
        res = ser - dt
        tm.assert_series_equal(res, expected)
        res = dt - ser
        tm.assert_series_equal(res, -expected)

    def test_datetimeindex_sub_timestamp_overflow(self):
        dtimax = pd.to_datetime(['2021-12-28 17:19', Timestamp.max]).as_unit('ns')
        dtimin = pd.to_datetime(['2021-12-28 17:19', Timestamp.min]).as_unit('ns')
        tsneg = Timestamp('1950-01-01').as_unit('ns')
        ts_neg_variants = [tsneg, tsneg.to_pydatetime(), tsneg.to_datetime64().astype('datetime64[ns]'), tsneg.to_datetime64().astype('datetime64[D]')]
        tspos = Timestamp('1980-01-01').as_unit('ns')
        ts_pos_variants = [tspos, tspos.to_pydatetime(), tspos.to_datetime64().astype('datetime64[ns]'), tspos.to_datetime64().astype('datetime64[D]')]
        msg = 'Overflow in int64 addition'
        for variant in ts_neg_variants:
            with pytest.raises(OverflowError, match=msg):
                dtimax - variant
        expected = Timestamp.max._value - tspos._value
        for variant in ts_pos_variants:
            res = dtimax - variant
            assert res[1]._value == expected
        expected = Timestamp.min._value - tsneg._value
        for variant in ts_neg_variants:
            res = dtimin - variant
            assert res[1]._value == expected
        for variant in ts_pos_variants:
            with pytest.raises(OverflowError, match=msg):
                dtimin - variant

    def test_datetimeindex_sub_datetimeindex_overflow(self):
        dtimax = pd.to_datetime(['2021-12-28 17:19', Timestamp.max]).as_unit('ns')
        dtimin = pd.to_datetime(['2021-12-28 17:19', Timestamp.min]).as_unit('ns')
        ts_neg = pd.to_datetime(['1950-01-01', '1950-01-01']).as_unit('ns')
        ts_pos = pd.to_datetime(['1980-01-01', '1980-01-01']).as_unit('ns')
        expected = Timestamp.max._value - ts_pos[1]._value
        result = dtimax - ts_pos
        assert result[1]._value == expected
        expected = Timestamp.min._value - ts_neg[1]._value
        result = dtimin - ts_neg
        assert result[1]._value == expected
        msg = 'Overflow in int64 addition'
        with pytest.raises(OverflowError, match=msg):
            dtimax - ts_neg
        with pytest.raises(OverflowError, match=msg):
            dtimin - ts_pos
        tmin = pd.to_datetime([Timestamp.min])
        t1 = tmin + Timedelta.max + Timedelta('1us')
        with pytest.raises(OverflowError, match=msg):
            t1 - tmin
        tmax = pd.to_datetime([Timestamp.max])
        t2 = tmax + Timedelta.min - Timedelta('1us')
        with pytest.raises(OverflowError, match=msg):
            tmax - t2