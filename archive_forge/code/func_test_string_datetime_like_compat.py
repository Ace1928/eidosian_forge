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
def test_string_datetime_like_compat():
    data = ['2004-01', '2004-02', '2004-03', '2004-04']
    expected = frequencies.infer_freq(data)
    result = frequencies.infer_freq(Index(data))
    assert result == expected