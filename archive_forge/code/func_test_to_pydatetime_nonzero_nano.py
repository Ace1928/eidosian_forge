from datetime import (
import pytz
from pandas._libs.tslibs.timezones import dateutil_gettz as gettz
import pandas.util._test_decorators as td
from pandas import Timestamp
import pandas._testing as tm
def test_to_pydatetime_nonzero_nano(self):
    ts = Timestamp('2011-01-01 9:00:00.123456789')
    with tm.assert_produces_warning(UserWarning):
        expected = datetime(2011, 1, 1, 9, 0, 0, 123456)
        result = ts.to_pydatetime()
        assert result == expected