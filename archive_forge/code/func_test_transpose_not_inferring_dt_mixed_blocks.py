import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_transpose_not_inferring_dt_mixed_blocks(self):
    df = DataFrame({'a': Series([Timestamp('2019-12-31'), Timestamp('2019-12-31')], dtype=object), 'b': [Timestamp('2019-12-31'), Timestamp('2019-12-31')]})
    result = df.T
    expected = DataFrame([[Timestamp('2019-12-31'), Timestamp('2019-12-31')], [Timestamp('2019-12-31'), Timestamp('2019-12-31')]], columns=[0, 1], index=['a', 'b'], dtype=object)
    tm.assert_frame_equal(result, expected)