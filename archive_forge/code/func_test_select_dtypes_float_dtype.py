import numpy as np
import pytest
from pandas.core.dtypes.dtypes import ExtensionDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import ExtensionArray
@pytest.mark.parametrize('expected, float_dtypes', [[DataFrame({'A': range(3), 'B': range(5, 8), 'C': range(10, 7, -1)}).astype(dtype={'A': float, 'B': np.float64, 'C': np.float32}), float], [DataFrame({'A': range(3), 'B': range(5, 8), 'C': range(10, 7, -1)}).astype(dtype={'A': float, 'B': np.float64, 'C': np.float32}), 'float'], [DataFrame({'C': range(10, 7, -1)}, dtype=np.float32), np.float32], [DataFrame({'A': range(3), 'B': range(5, 8)}).astype(dtype={'A': float, 'B': np.float64}), np.float64]])
def test_select_dtypes_float_dtype(self, expected, float_dtypes):
    dtype_dict = {'A': float, 'B': np.float64, 'C': np.float32}
    df = DataFrame({'A': range(3), 'B': range(5, 8), 'C': range(10, 7, -1)})
    df = df.astype(dtype_dict)
    result = df.select_dtypes(include=float_dtypes)
    tm.assert_frame_equal(result, expected)