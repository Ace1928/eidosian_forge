from copy import (
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
from pandas import (
import pandas._testing as tm
def test_squeeze_axis_len_3(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((3, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=3, freq='B'))
    tm.assert_frame_equal(df.squeeze(axis=0), df)