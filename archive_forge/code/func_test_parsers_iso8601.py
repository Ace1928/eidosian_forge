from datetime import datetime
import pytest
from pandas._libs import tslib
from pandas import Timestamp
@pytest.mark.parametrize('date_str, exp', [('2011-01-02', datetime(2011, 1, 2)), ('2011-1-2', datetime(2011, 1, 2)), ('2011-01', datetime(2011, 1, 1)), ('2011-1', datetime(2011, 1, 1)), ('2011 01 02', datetime(2011, 1, 2)), ('2011.01.02', datetime(2011, 1, 2)), ('2011/01/02', datetime(2011, 1, 2)), ('2011\\01\\02', datetime(2011, 1, 2)), ('2013-01-01 05:30:00', datetime(2013, 1, 1, 5, 30)), ('2013-1-1 5:30:00', datetime(2013, 1, 1, 5, 30)), ('2013-1-1 5:30:00+01:00', Timestamp(2013, 1, 1, 5, 30, tz='UTC+01:00'))])
def test_parsers_iso8601(date_str, exp):
    actual = tslib._test_parse_iso8601(date_str)
    assert actual == exp