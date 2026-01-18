import numpy as np
from pandas.core.dtypes.common import is_scalar
import pandas as pd
import pandas._testing as tm
def test_searchsorted_string(self, string_dtype):
    arr = pd.array(['a', 'b', 'c'], dtype=string_dtype)
    result = arr.searchsorted('a', side='left')
    assert is_scalar(result)
    assert result == 0
    result = arr.searchsorted('a', side='right')
    assert is_scalar(result)
    assert result == 1