import numpy as np
import pytest
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
def test_numpy_array_all_dtypes(any_numpy_dtype):
    ser = Series(dtype=any_numpy_dtype)
    result = ser.array
    if np.dtype(any_numpy_dtype).kind == 'M':
        assert isinstance(result, DatetimeArray)
    elif np.dtype(any_numpy_dtype).kind == 'm':
        assert isinstance(result, TimedeltaArray)
    else:
        assert isinstance(result, NumpyExtensionArray)