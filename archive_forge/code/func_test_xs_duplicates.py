import re
import numpy as np
import pytest
from pandas.errors import SettingWithCopyError
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
def test_xs_duplicates(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 2)), index=['b', 'b', 'c', 'b', 'a'])
    cross = df.xs('c')
    exp = df.iloc[2]
    tm.assert_series_equal(cross, exp)