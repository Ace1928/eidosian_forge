import numpy as np
import pytest
from pandas.core.dtypes.dtypes import ExtensionDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import ExtensionArray
def test_select_dtypes_numeric_nullable_string(self, nullable_string_dtype):
    arr = pd.array(['a', 'b'], dtype=nullable_string_dtype)
    df = DataFrame(arr)
    is_selected = df.select_dtypes(np.number).shape == df.shape
    assert not is_selected