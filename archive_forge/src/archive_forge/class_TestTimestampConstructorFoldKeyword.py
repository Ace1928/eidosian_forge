import calendar
from datetime import (
import zoneinfo
import dateutil.tz
from dateutil.tz import (
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.compat import PY310
from pandas.errors import OutOfBoundsDatetime
from pandas import (
class TestTimestampConstructorFoldKeyword:

    def test_timestamp_constructor_invalid_fold_raise(self):
        msg = 'Valid values for the fold argument are None, 0, or 1.'
        with pytest.raises(ValueError, match=msg):
            Timestamp(123, fold=2)

    def test_timestamp_constructor_pytz_fold_raise(self):
        msg = 'pytz timezones do not support fold. Please use dateutil timezones.'
        tz = pytz.timezone('Europe/London')
        with pytest.raises(ValueError, match=msg):
            Timestamp(datetime(2019, 10, 27, 0, 30, 0, 0), tz=tz, fold=0)

    @pytest.mark.parametrize('fold', [0, 1])
    @pytest.mark.parametrize('ts_input', [1572136200000000000, 1.5721362e+18, np.datetime64(1572136200000000000, 'ns'), '2019-10-27 01:30:00+01:00', datetime(2019, 10, 27, 0, 30, 0, 0, tzinfo=timezone.utc)])
    def test_timestamp_constructor_fold_conflict(self, ts_input, fold):
        msg = 'Cannot pass fold with possibly unambiguous input: int, float, numpy.datetime64, str, or timezone-aware datetime-like. Pass naive datetime-like or build Timestamp from components.'
        with pytest.raises(ValueError, match=msg):
            Timestamp(ts_input=ts_input, fold=fold)

    @pytest.mark.parametrize('tz', ['dateutil/Europe/London', None])
    @pytest.mark.parametrize('fold', [0, 1])
    def test_timestamp_constructor_retain_fold(self, tz, fold):
        ts = Timestamp(year=2019, month=10, day=27, hour=1, minute=30, tz=tz, fold=fold)
        result = ts.fold
        expected = fold
        assert result == expected
    try:
        _tzs = ['dateutil/Europe/London', zoneinfo.ZoneInfo('Europe/London')]
    except zoneinfo.ZoneInfoNotFoundError:
        _tzs = ['dateutil/Europe/London']

    @pytest.mark.parametrize('tz', _tzs)
    @pytest.mark.parametrize('ts_input,fold_out', [(1572136200000000000, 0), (1572139800000000000, 1), ('2019-10-27 01:30:00+01:00', 0), ('2019-10-27 01:30:00+00:00', 1), (datetime(2019, 10, 27, 1, 30, 0, 0, fold=0), 0), (datetime(2019, 10, 27, 1, 30, 0, 0, fold=1), 1)])
    def test_timestamp_constructor_infer_fold_from_value(self, tz, ts_input, fold_out):
        ts = Timestamp(ts_input, tz=tz)
        result = ts.fold
        expected = fold_out
        assert result == expected

    @pytest.mark.parametrize('tz', ['dateutil/Europe/London'])
    @pytest.mark.parametrize('ts_input,fold,value_out', [(datetime(2019, 10, 27, 1, 30, 0, 0), 0, 1572136200000000), (datetime(2019, 10, 27, 1, 30, 0, 0), 1, 1572139800000000)])
    def test_timestamp_constructor_adjust_value_for_fold(self, tz, ts_input, fold, value_out):
        ts = Timestamp(ts_input, tz=tz, fold=fold)
        result = ts._value
        expected = value_out
        assert result == expected