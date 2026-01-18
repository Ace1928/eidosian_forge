from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
def test_subtraction_ops_with_tz(self, box_with_array):
    dti = pd.date_range('20130101', periods=3)
    dti = tm.box_expected(dti, box_with_array)
    ts = Timestamp('20130101')
    dt = ts.to_pydatetime()
    dti_tz = pd.date_range('20130101', periods=3).tz_localize('US/Eastern')
    dti_tz = tm.box_expected(dti_tz, box_with_array)
    ts_tz = Timestamp('20130101').tz_localize('US/Eastern')
    ts_tz2 = Timestamp('20130101').tz_localize('CET')
    dt_tz = ts_tz.to_pydatetime()
    td = Timedelta('1 days')

    def _check(result, expected):
        assert result == expected
        assert isinstance(result, Timedelta)
    result = ts - ts
    expected = Timedelta('0 days')
    _check(result, expected)
    result = dt_tz - ts_tz
    expected = Timedelta('0 days')
    _check(result, expected)
    result = ts_tz - dt_tz
    expected = Timedelta('0 days')
    _check(result, expected)
    msg = 'Cannot subtract tz-naive and tz-aware datetime-like objects.'
    with pytest.raises(TypeError, match=msg):
        dt_tz - ts
    msg = "can't subtract offset-naive and offset-aware datetimes"
    with pytest.raises(TypeError, match=msg):
        dt_tz - dt
    msg = "can't subtract offset-naive and offset-aware datetimes"
    with pytest.raises(TypeError, match=msg):
        dt - dt_tz
    msg = 'Cannot subtract tz-naive and tz-aware datetime-like objects.'
    with pytest.raises(TypeError, match=msg):
        ts - dt_tz
    with pytest.raises(TypeError, match=msg):
        ts_tz2 - ts
    with pytest.raises(TypeError, match=msg):
        ts_tz2 - dt
    msg = 'Cannot subtract tz-naive and tz-aware'
    with pytest.raises(TypeError, match=msg):
        dti - ts_tz
    with pytest.raises(TypeError, match=msg):
        dti_tz - ts
    result = dti_tz - dt_tz
    expected = TimedeltaIndex(['0 days', '1 days', '2 days'])
    expected = tm.box_expected(expected, box_with_array)
    tm.assert_equal(result, expected)
    result = dt_tz - dti_tz
    expected = TimedeltaIndex(['0 days', '-1 days', '-2 days'])
    expected = tm.box_expected(expected, box_with_array)
    tm.assert_equal(result, expected)
    result = dti_tz - ts_tz
    expected = TimedeltaIndex(['0 days', '1 days', '2 days'])
    expected = tm.box_expected(expected, box_with_array)
    tm.assert_equal(result, expected)
    result = ts_tz - dti_tz
    expected = TimedeltaIndex(['0 days', '-1 days', '-2 days'])
    expected = tm.box_expected(expected, box_with_array)
    tm.assert_equal(result, expected)
    result = td - td
    expected = Timedelta('0 days')
    _check(result, expected)
    result = dti_tz - td
    expected = DatetimeIndex(['20121231', '20130101', '20130102'], tz='US/Eastern').as_unit('ns')
    expected = tm.box_expected(expected, box_with_array)
    tm.assert_equal(result, expected)