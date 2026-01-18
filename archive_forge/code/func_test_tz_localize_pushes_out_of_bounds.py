from datetime import timedelta
import re
from dateutil.tz import gettz
import pytest
import pytz
from pytz.exceptions import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.errors import OutOfBoundsDatetime
from pandas import (
@pytest.mark.skip_ubsan
def test_tz_localize_pushes_out_of_bounds(self):
    msg = f'Converting {Timestamp.min.strftime('%Y-%m-%d %H:%M:%S')} underflows past {Timestamp.min}'
    pac = Timestamp.min.tz_localize('US/Pacific')
    assert pac._value > Timestamp.min._value
    pac.tz_convert('Asia/Tokyo')
    with pytest.raises(OutOfBoundsDatetime, match=msg):
        Timestamp.min.tz_localize('Asia/Tokyo')
    msg = f'Converting {Timestamp.max.strftime('%Y-%m-%d %H:%M:%S')} overflows past {Timestamp.max}'
    tokyo = Timestamp.max.tz_localize('Asia/Tokyo')
    assert tokyo._value < Timestamp.max._value
    tokyo.tz_convert('US/Pacific')
    with pytest.raises(OutOfBoundsDatetime, match=msg):
        Timestamp.max.tz_localize('US/Pacific')