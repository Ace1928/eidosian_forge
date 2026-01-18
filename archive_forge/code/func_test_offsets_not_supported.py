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
@pytest.mark.parametrize('freq, freq_msg', [(offsets.BYearBegin(), 'BYearBegin'), (offsets.YearBegin(2), 'YearBegin'), (offsets.QuarterBegin(startingMonth=12), 'QuarterBegin'), (offsets.BusinessMonthEnd(2), 'BusinessMonthEnd')])
def test_offsets_not_supported(self, freq, freq_msg):
    msg = re.escape(f'{freq} is not supported as period frequency')
    with pytest.raises(ValueError, match=msg):
        Period(year=2014, freq=freq)