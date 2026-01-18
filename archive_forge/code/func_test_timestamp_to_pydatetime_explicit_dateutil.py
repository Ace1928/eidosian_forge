from datetime import (
import pytz
from pandas._libs.tslibs.timezones import dateutil_gettz as gettz
import pandas.util._test_decorators as td
from pandas import Timestamp
import pandas._testing as tm
@td.skip_if_windows
def test_timestamp_to_pydatetime_explicit_dateutil(self):
    stamp = Timestamp('20090415', tz=gettz('US/Eastern'))
    dtval = stamp.to_pydatetime()
    assert stamp == dtval
    assert stamp.tzinfo == dtval.tzinfo