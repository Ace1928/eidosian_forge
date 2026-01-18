import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_transpose_td64_intervals(self):
    tdi = timedelta_range('0 Days', '3 Days')
    ii = IntervalIndex.from_breaks(tdi)
    ii = ii.insert(-1, np.nan)
    df = DataFrame(ii)
    result = df.T
    expected = DataFrame({i: ii[i:i + 1] for i in range(len(ii))})
    tm.assert_frame_equal(result, expected)