import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_shift_named_axis(self):
    df = DataFrame(np.random.default_rng(2).random((10, 5)))
    expected = pd.concat([DataFrame(np.nan, index=df.index, columns=[0]), df.iloc[:, 0:-1]], ignore_index=True, axis=1)
    result = df.shift(1, axis='columns')
    tm.assert_frame_equal(result, expected)