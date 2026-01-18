import numpy as np
import pytest
from pandas.core.dtypes.dtypes import NumpyEADtype
import pandas as pd
import pandas._testing as tm
from pandas.arrays import NumpyExtensionArray
def test_setitem_no_coercion():
    arr = NumpyExtensionArray(np.array([1, 2, 3]))
    with pytest.raises(ValueError, match='int'):
        arr[0] = 'a'
    arr[0] = 2.5
    assert isinstance(arr[0], (int, np.integer)), type(arr[0])