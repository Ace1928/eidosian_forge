import calendar
from datetime import (
import locale
import time
import unicodedata
from dateutil.tz import (
from hypothesis import (
import numpy as np
import pytest
import pytz
from pytz import utc
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas._libs.tslibs.timezones import (
from pandas.compat import IS64
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
class TestTimestamp:

    @pytest.mark.parametrize('tz', [None, pytz.timezone('US/Pacific')])
    def test_disallow_setting_tz(self, tz):
        ts = Timestamp('2010')
        msg = 'Cannot directly set timezone'
        with pytest.raises(AttributeError, match=msg):
            ts.tz = tz

    def test_default_to_stdlib_utc(self):
        assert Timestamp.utcnow().tz is timezone.utc
        assert Timestamp.now('UTC').tz is timezone.utc
        assert Timestamp('2016-01-01', tz='UTC').tz is timezone.utc

    def test_tz(self):
        tstr = '2014-02-01 09:00'
        ts = Timestamp(tstr)
        local = ts.tz_localize('Asia/Tokyo')
        assert local.hour == 9
        assert local == Timestamp(tstr, tz='Asia/Tokyo')
        conv = local.tz_convert('US/Eastern')
        assert conv == Timestamp('2014-01-31 19:00', tz='US/Eastern')
        assert conv.hour == 19
        ts = Timestamp(tstr) + offsets.Nano(5)
        local = ts.tz_localize('Asia/Tokyo')
        assert local.hour == 9
        assert local.nanosecond == 5
        conv = local.tz_convert('US/Eastern')
        assert conv.nanosecond == 5
        assert conv.hour == 19

    def test_utc_z_designator(self):
        assert get_timezone(Timestamp('2014-11-02 01:00Z').tzinfo) is timezone.utc

    def test_asm8(self):
        ns = [Timestamp.min._value, Timestamp.max._value, 1000]
        for n in ns:
            assert Timestamp(n).asm8.view('i8') == np.datetime64(n, 'ns').view('i8') == n
        assert Timestamp('nat').asm8.view('i8') == np.datetime64('nat', 'ns').view('i8')

    def test_class_ops(self):

        def compare(x, y):
            assert int((Timestamp(x)._value - Timestamp(y)._value) / 1000000000.0) == 0
        compare(Timestamp.now(), datetime.now())
        compare(Timestamp.now('UTC'), datetime.now(pytz.timezone('UTC')))
        compare(Timestamp.now('UTC'), datetime.now(tzutc()))
        compare(Timestamp.utcnow(), datetime.now(timezone.utc))
        compare(Timestamp.today(), datetime.today())
        current_time = calendar.timegm(datetime.now().utctimetuple())
        ts_utc = Timestamp.utcfromtimestamp(current_time)
        assert ts_utc.timestamp() == current_time
        compare(Timestamp.fromtimestamp(current_time), datetime.fromtimestamp(current_time))
        compare(Timestamp.fromtimestamp(current_time, 'UTC'), datetime.fromtimestamp(current_time, utc))
        compare(Timestamp.fromtimestamp(current_time, tz='UTC'), datetime.fromtimestamp(current_time, utc))
        date_component = datetime.now(timezone.utc)
        time_component = (date_component + timedelta(minutes=10)).time()
        compare(Timestamp.combine(date_component, time_component), datetime.combine(date_component, time_component))

    def test_basics_nanos(self):
        val = np.int64(946684800000000000).view('M8[ns]')
        stamp = Timestamp(val.view('i8') + 500)
        assert stamp.year == 2000
        assert stamp.month == 1
        assert stamp.microsecond == 0
        assert stamp.nanosecond == 500
        val = np.iinfo(np.int64).min + 80000000000000
        stamp = Timestamp(val)
        assert stamp.year == 1677
        assert stamp.month == 9
        assert stamp.day == 21
        assert stamp.microsecond == 145224
        assert stamp.nanosecond == 192

    def test_roundtrip(self):
        base = Timestamp('20140101 00:00:00').as_unit('ns')
        result = Timestamp(base._value + Timedelta('5ms')._value)
        assert result == Timestamp(f'{base}.005000')
        assert result.microsecond == 5000
        result = Timestamp(base._value + Timedelta('5us')._value)
        assert result == Timestamp(f'{base}.000005')
        assert result.microsecond == 5
        result = Timestamp(base._value + Timedelta('5ns')._value)
        assert result == Timestamp(f'{base}.000000005')
        assert result.nanosecond == 5
        assert result.microsecond == 0
        result = Timestamp(base._value + Timedelta('6ms 5us')._value)
        assert result == Timestamp(f'{base}.006005')
        assert result.microsecond == 5 + 6 * 1000
        result = Timestamp(base._value + Timedelta('200ms 5us')._value)
        assert result == Timestamp(f'{base}.200005')
        assert result.microsecond == 5 + 200 * 1000

    def test_hash_equivalent(self):
        d = {datetime(2011, 1, 1): 5}
        stamp = Timestamp(datetime(2011, 1, 1))
        assert d[stamp] == 5

    @pytest.mark.parametrize('timezone, year, month, day, hour', [['America/Chicago', 2013, 11, 3, 1], ['America/Santiago', 2021, 4, 3, 23]])
    def test_hash_timestamp_with_fold(self, timezone, year, month, day, hour):
        test_timezone = gettz(timezone)
        transition_1 = Timestamp(year=year, month=month, day=day, hour=hour, minute=0, fold=0, tzinfo=test_timezone)
        transition_2 = Timestamp(year=year, month=month, day=day, hour=hour, minute=0, fold=1, tzinfo=test_timezone)
        assert hash(transition_1) == hash(transition_2)