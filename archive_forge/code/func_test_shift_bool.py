import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_shift_bool(self):
    df = DataFrame({'high': [True, False], 'low': [False, False]})
    rs = df.shift(1)
    xp = DataFrame(np.array([[np.nan, np.nan], [True, False]], dtype=object), columns=['high', 'low'])
    tm.assert_frame_equal(rs, xp)