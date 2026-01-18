import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_xs_level_series(self, multiindex_dataframe_random_data):
    df = multiindex_dataframe_random_data
    ser = df['A']
    expected = ser[:, 'two']
    result = df.xs('two', level=1)['A']
    tm.assert_series_equal(result, expected)