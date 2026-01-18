from datetime import (
import re
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas._libs.tslibs.ccalendar import (
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from pandas._libs.tslibs.parsing import DateParseError
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas import (
import pandas._testing as tm
def test_to_timestamp_business_end(self):
    with tm.assert_produces_warning(FutureWarning, match=bday_msg):
        per = Period('1990-01-05', 'B')
        result = per.to_timestamp('B', how='E')
    expected = Timestamp('1990-01-06') - Timedelta(nanoseconds=1)
    assert result == expected