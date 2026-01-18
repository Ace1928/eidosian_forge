import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
import pandas._testing as tm
@pytest.mark.xfail(using_pyarrow_string_dtype(), reason='repr different')
def test_repr_floats(self):
    markers = Series(['foo', 'bar'], index=IntervalIndex([Interval(left, right) for left, right in zip(Index([329.973, 345.137], dtype='float64'), Index([345.137, 360.191], dtype='float64'))]))
    result = str(markers)
    expected = '(329.973, 345.137]    foo\n(345.137, 360.191]    bar\ndtype: object'
    assert result == expected