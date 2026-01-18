import numpy as np
import pytest
import pandas as pd
from pandas import Series
import pandas._testing as tm
def test_validate_median_initial():
    ser = Series([1, 2])
    msg = "the 'overwrite_input' parameter is not supported in the pandas implementation of median\\(\\)"
    with pytest.raises(ValueError, match=msg):
        ser.median(overwrite_input=True)