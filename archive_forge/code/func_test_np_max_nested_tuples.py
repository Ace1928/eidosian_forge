import numpy as np
import pytest
from pandas.core.dtypes.dtypes import NumpyEADtype
import pandas as pd
import pandas._testing as tm
from pandas.arrays import NumpyExtensionArray
def test_np_max_nested_tuples():
    vals = [(('j', 'k'), ('l', 'm')), (('l', 'm'), ('o', 'p')), (('o', 'p'), ('j', 'k'))]
    ser = pd.Series(vals)
    arr = ser.array
    assert arr.max() is arr[2]
    assert ser.max() is arr[2]
    result = np.maximum.reduce(arr)
    assert result == arr[2]
    result = np.maximum.reduce(ser)
    assert result == arr[2]