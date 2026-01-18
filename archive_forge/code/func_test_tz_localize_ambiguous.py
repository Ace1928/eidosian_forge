from datetime import timedelta
import re
from dateutil.tz import gettz
import pytest
import pytz
from pytz.exceptions import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.errors import OutOfBoundsDatetime
from pandas import (
def test_tz_localize_ambiguous(self):
    ts = Timestamp('2014-11-02 01:00')
    ts_dst = ts.tz_localize('US/Eastern', ambiguous=True)
    ts_no_dst = ts.tz_localize('US/Eastern', ambiguous=False)
    assert ts_no_dst._value - ts_dst._value == 3600
    msg = re.escape("'ambiguous' parameter must be one of: True, False, 'NaT', 'raise' (default)")
    with pytest.raises(ValueError, match=msg):
        ts.tz_localize('US/Eastern', ambiguous='infer')
    msg = 'Cannot localize tz-aware Timestamp, use tz_convert for conversions'
    with pytest.raises(TypeError, match=msg):
        Timestamp('2011-01-01', tz='US/Eastern').tz_localize('Asia/Tokyo')
    msg = 'Cannot convert tz-naive Timestamp, use tz_localize to localize'
    with pytest.raises(TypeError, match=msg):
        Timestamp('2011-01-01').tz_convert('Asia/Tokyo')