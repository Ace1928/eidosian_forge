from datetime import (
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import (
def test_union_bug_1745(self, sort):
    left = DatetimeIndex(['2012-05-11 15:19:49.695000'])
    right = DatetimeIndex(['2012-05-29 13:04:21.322000', '2012-05-11 15:27:24.873000', '2012-05-11 15:31:05.350000'])
    result = left.union(right, sort=sort)
    exp = DatetimeIndex(['2012-05-11 15:19:49.695000', '2012-05-29 13:04:21.322000', '2012-05-11 15:27:24.873000', '2012-05-11 15:31:05.350000'])
    if sort is None:
        exp = exp.sort_values()
    tm.assert_index_equal(result, exp)