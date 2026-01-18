import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('input_data, output_data', [(np.empty(shape=(0,)), []), (np.ones(shape=(2,)), [np.nan, 1.0])])
def test_shift_non_writable_array(self, input_data, output_data, frame_or_series):
    input_data.setflags(write=False)
    result = frame_or_series(input_data).shift(1)
    if frame_or_series is not Series:
        expected = frame_or_series(output_data, index=range(len(output_data)), columns=range(1), dtype='float64')
    else:
        expected = frame_or_series(output_data, dtype='float64')
    tm.assert_equal(result, expected)