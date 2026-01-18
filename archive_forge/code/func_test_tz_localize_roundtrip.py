from datetime import timedelta
import re
from dateutil.tz import gettz
import pytest
import pytz
from pytz.exceptions import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.errors import OutOfBoundsDatetime
from pandas import (
@pytest.mark.parametrize('stamp', ['2014-02-01 09:00', '2014-07-08 09:00', '2014-11-01 17:00', '2014-11-05 00:00'])
def test_tz_localize_roundtrip(self, stamp, tz_aware_fixture):
    tz = tz_aware_fixture
    ts = Timestamp(stamp)
    localized = ts.tz_localize(tz)
    assert localized == Timestamp(stamp, tz=tz)
    msg = 'Cannot localize tz-aware Timestamp'
    with pytest.raises(TypeError, match=msg):
        localized.tz_localize(tz)
    reset = localized.tz_localize(None)
    assert reset == ts
    assert reset.tzinfo is None