import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_vars_work_with_multiindex(self, df1):
    expected = DataFrame({('A', 'a'): df1['A', 'a'], 'CAP': ['B'] * len(df1), 'low': ['b'] * len(df1), 'value': df1['B', 'b']}, columns=[('A', 'a'), 'CAP', 'low', 'value'])
    result = df1.melt(id_vars=[('A', 'a')], value_vars=[('B', 'b')])
    tm.assert_frame_equal(result, expected)