import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_shift_with_iterable_series(self):
    data = {'a': [1, 2, 3]}
    shifts = [0, 1, 2]
    df = DataFrame(data)
    s = df['a']
    tm.assert_frame_equal(s.shift(shifts), df.shift(shifts))