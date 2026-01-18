from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
def test_timedelta_ops(self):
    s = Series([Timestamp('20130101') + timedelta(seconds=i * i) for i in range(10)])
    td = s.diff()
    result = td.mean()
    expected = to_timedelta(timedelta(seconds=9))
    assert result == expected
    result = td.to_frame().mean()
    assert result[0] == expected
    result = td.quantile(0.1)
    expected = Timedelta(np.timedelta64(2600, 'ms'))
    assert result == expected
    result = td.median()
    expected = to_timedelta('00:00:09')
    assert result == expected
    result = td.to_frame().median()
    assert result[0] == expected
    result = td.sum()
    expected = to_timedelta('00:01:21')
    assert result == expected
    result = td.to_frame().sum()
    assert result[0] == expected
    result = td.std()
    expected = to_timedelta(Series(td.dropna().values).std())
    assert result == expected
    result = td.to_frame().std()
    assert result[0] == expected
    s = Series([Timestamp('2015-02-03'), Timestamp('2015-02-07')])
    assert s.diff().median() == timedelta(days=4)
    s = Series([Timestamp('2015-02-03'), Timestamp('2015-02-07'), Timestamp('2015-02-15')])
    assert s.diff().median() == timedelta(days=6)