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
class TestPeriodComparisons:

    def test_sort_periods(self):
        jan = Period('2000-01', 'M')
        feb = Period('2000-02', 'M')
        mar = Period('2000-03', 'M')
        periods = [mar, jan, feb]
        correctPeriods = [jan, feb, mar]
        assert sorted(periods) == correctPeriods