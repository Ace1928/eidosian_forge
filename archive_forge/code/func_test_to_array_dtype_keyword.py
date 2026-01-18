import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import FloatingArray
from pandas.core.arrays.floating import (
def test_to_array_dtype_keyword():
    result = pd.array([1, 2], dtype='Float32')
    assert result.dtype == Float32Dtype()
    result = pd.array(np.array([1, 2], dtype='float32'), dtype='Float64')
    assert result.dtype == Float64Dtype()