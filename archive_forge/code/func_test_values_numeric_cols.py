import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_values_numeric_cols(self, float_frame):
    float_frame['foo'] = 'bar'
    values = float_frame[['A', 'B', 'C', 'D']].values
    assert values.dtype == np.float64