from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.ccalendar import (
from pandas._libs.tslibs.offsets import _get_offset
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas.compat import is_platform_windows
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.tools.datetimes import to_datetime
from pandas.tseries import (
@pytest.mark.parametrize('freq', [None, 'ms'])
def test_series_period_index(freq):
    msg = 'cannot infer freq from a non-convertible dtype on a Series'
    s = Series(period_range('2013', periods=10, freq=freq))
    with pytest.raises(TypeError, match=msg):
        frequencies.infer_freq(s)