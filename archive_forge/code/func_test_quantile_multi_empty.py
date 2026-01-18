import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_quantile_multi_empty(self, interp_method):
    interpolation, method = interp_method
    result = DataFrame({'x': [], 'y': []}).quantile([0.1, 0.9], axis=0, interpolation=interpolation, method=method)
    expected = DataFrame({'x': [np.nan, np.nan], 'y': [np.nan, np.nan]}, index=[0.1, 0.9])
    tm.assert_frame_equal(result, expected)