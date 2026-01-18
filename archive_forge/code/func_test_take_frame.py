from copy import (
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
from pandas import (
import pandas._testing as tm
def test_take_frame(self):
    indices = [1, 5, -2, 6, 3, -1]
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
    out = df.take(indices)
    expected = DataFrame(data=df.values.take(indices, axis=0), index=df.index.take(indices), columns=df.columns)
    tm.assert_frame_equal(out, expected)