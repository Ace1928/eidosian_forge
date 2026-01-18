import operator
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.arrays import TimedeltaArray
from pandas.tests.arithmetic.common import (
def test_pi_add_sub_td64_array_tick(self):
    rng = period_range('1/1/2000', freq='90D', periods=3)
    tdi = TimedeltaIndex(['-1 Day', '-1 Day', '-1 Day'])
    tdarr = tdi.values
    expected = period_range('12/31/1999', freq='90D', periods=3)
    result = rng + tdi
    tm.assert_index_equal(result, expected)
    result = rng + tdarr
    tm.assert_index_equal(result, expected)
    result = tdi + rng
    tm.assert_index_equal(result, expected)
    result = tdarr + rng
    tm.assert_index_equal(result, expected)
    expected = period_range('1/2/2000', freq='90D', periods=3)
    result = rng - tdi
    tm.assert_index_equal(result, expected)
    result = rng - tdarr
    tm.assert_index_equal(result, expected)
    msg = 'cannot subtract .* from .*'
    with pytest.raises(TypeError, match=msg):
        tdarr - rng
    with pytest.raises(TypeError, match=msg):
        tdi - rng