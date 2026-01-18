import numpy as np
import pytest
import pandas as pd
from pandas import Series
import pandas._testing as tm
def test_validate_stat_keepdims():
    ser = Series([1, 2])
    msg = "the 'keepdims' parameter is not supported in the pandas implementation of sum\\(\\)"
    with pytest.raises(ValueError, match=msg):
        np.sum(ser, keepdims=True)