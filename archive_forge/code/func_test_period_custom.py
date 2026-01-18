from contextlib import nullcontext
from datetime import (
import locale
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_period_custom(self):
    msg = 'PeriodIndex.format is deprecated'
    per = pd.period_range('2003-01-01 12:01:01.123', periods=2, freq='ms')
    with tm.assert_produces_warning(FutureWarning, match=msg):
        formatted = per.format(date_format='%y %I:%M:%S (ms=%l us=%u ns=%n)')
    assert formatted[0] == '03 12:01:01 (ms=123 us=123000 ns=123000000)'
    assert formatted[1] == '03 12:01:01 (ms=124 us=124000 ns=124000000)'
    per = pd.period_range('2003-01-01 12:01:01.123456', periods=2, freq='us')
    with tm.assert_produces_warning(FutureWarning, match=msg):
        formatted = per.format(date_format='%y %I:%M:%S (ms=%l us=%u ns=%n)')
    assert formatted[0] == '03 12:01:01 (ms=123 us=123456 ns=123456000)'
    assert formatted[1] == '03 12:01:01 (ms=123 us=123457 ns=123457000)'
    per = pd.period_range('2003-01-01 12:01:01.123456789', periods=2, freq='ns')
    with tm.assert_produces_warning(FutureWarning, match=msg):
        formatted = per.format(date_format='%y %I:%M:%S (ms=%l us=%u ns=%n)')
    assert formatted[0] == '03 12:01:01 (ms=123 us=123456 ns=123456789)'
    assert formatted[1] == '03 12:01:01 (ms=123 us=123456 ns=123456790)'