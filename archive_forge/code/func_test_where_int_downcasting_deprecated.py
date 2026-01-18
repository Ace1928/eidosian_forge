from datetime import datetime
from hypothesis import given
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas._testing._hypothesis import OPTIONAL_ONE_OF_ALL
def test_where_int_downcasting_deprecated():
    arr = np.arange(6).astype(np.int16).reshape(3, 2)
    df = DataFrame(arr)
    mask = np.zeros(arr.shape, dtype=bool)
    mask[:, 0] = True
    res = df.where(mask, 2 ** 17)
    expected = DataFrame({0: arr[:, 0], 1: np.array([2 ** 17] * 3, dtype=np.int32)})
    tm.assert_frame_equal(res, expected)