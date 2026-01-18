from datetime import (
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import (
def test_union_bug_1730(self, sort):
    rng_a = date_range('1/1/2012', periods=4, freq='3h')
    rng_b = date_range('1/1/2012', periods=4, freq='4h')
    result = rng_a.union(rng_b, sort=sort)
    exp = list(rng_a) + list(rng_b[1:])
    if sort is None:
        exp = DatetimeIndex(sorted(exp))
    else:
        exp = DatetimeIndex(exp)
    tm.assert_index_equal(result, exp)