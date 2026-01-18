import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
def test_sort_values_no_op_reset_index(self):
    df = DataFrame({'A': [10, 20], 'B': [1, 5]}, index=[2, 3])
    result = df.sort_values(by='A', ignore_index=True)
    expected = DataFrame({'A': [10, 20], 'B': [1, 5]})
    tm.assert_frame_equal(result, expected)