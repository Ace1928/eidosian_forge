import numpy as np
import pytest
import pandas as pd
from pandas.core.arrays.floating import (
def test_repr_array():
    result = repr(pd.array([1.0, None, 3.0]))
    expected = '<FloatingArray>\n[1.0, <NA>, 3.0]\nLength: 3, dtype: Float64'
    assert result == expected