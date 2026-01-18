from datetime import datetime
import pytest
from pandas._libs import tslib
from pandas import Timestamp
@pytest.mark.parametrize('date_str', ['2011-01/02', '2011=11=11', '201401', '201111', '200101', '2005-0101', '200501-01', '20010101 12:3456', '20010101 1234:56', '20010101 1', '20010101 123', '20010101 12345', '20010101 12345Z'])
def test_parsers_iso8601_invalid(date_str):
    msg = f'Error parsing datetime string "{date_str}"'
    with pytest.raises(ValueError, match=msg):
        tslib._test_parse_iso8601(date_str)