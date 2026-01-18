from datetime import timezone
import pytest
import pytz
from pandas._libs.tslibs import timezones
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('method, exp', [['shift_forward', '2015-03-29 03:00:00'], ['shift_backward', '2015-03-29 01:59:59.999999999'], ['NaT', NaT], ['raise', None], ['foo', 'invalid']])
def test_tz_localize_nonexistent(self, warsaw, method, exp, unit):
    tz = warsaw
    n = 60
    dti = date_range(start='2015-03-29 02:00:00', periods=n, freq='min', unit=unit)
    ser = Series(1, index=dti)
    df = ser.to_frame()
    if method == 'raise':
        with tm.external_error_raised(pytz.NonExistentTimeError):
            dti.tz_localize(tz, nonexistent=method)
        with tm.external_error_raised(pytz.NonExistentTimeError):
            ser.tz_localize(tz, nonexistent=method)
        with tm.external_error_raised(pytz.NonExistentTimeError):
            df.tz_localize(tz, nonexistent=method)
    elif exp == 'invalid':
        msg = "The nonexistent argument must be one of 'raise', 'NaT', 'shift_forward', 'shift_backward' or a timedelta object"
        with pytest.raises(ValueError, match=msg):
            dti.tz_localize(tz, nonexistent=method)
        with pytest.raises(ValueError, match=msg):
            ser.tz_localize(tz, nonexistent=method)
        with pytest.raises(ValueError, match=msg):
            df.tz_localize(tz, nonexistent=method)
    else:
        result = ser.tz_localize(tz, nonexistent=method)
        expected = Series(1, index=DatetimeIndex([exp] * n, tz=tz).as_unit(unit))
        tm.assert_series_equal(result, expected)
        result = df.tz_localize(tz, nonexistent=method)
        expected = expected.to_frame()
        tm.assert_frame_equal(result, expected)
        res_index = dti.tz_localize(tz, nonexistent=method)
        tm.assert_index_equal(res_index, expected.index)