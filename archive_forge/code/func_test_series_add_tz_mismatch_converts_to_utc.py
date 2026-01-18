from datetime import (
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import IncompatibleFrequency
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_series_add_tz_mismatch_converts_to_utc(self):
    rng = date_range('1/1/2011', periods=100, freq='h', tz='utc')
    perm = np.random.default_rng(2).permutation(100)[:90]
    ser1 = Series(np.random.default_rng(2).standard_normal(90), index=rng.take(perm).tz_convert('US/Eastern'))
    perm = np.random.default_rng(2).permutation(100)[:90]
    ser2 = Series(np.random.default_rng(2).standard_normal(90), index=rng.take(perm).tz_convert('Europe/Berlin'))
    result = ser1 + ser2
    uts1 = ser1.tz_convert('utc')
    uts2 = ser2.tz_convert('utc')
    expected = uts1 + uts2
    expected = expected.sort_index()
    assert result.index.tz is timezone.utc
    tm.assert_series_equal(result, expected)