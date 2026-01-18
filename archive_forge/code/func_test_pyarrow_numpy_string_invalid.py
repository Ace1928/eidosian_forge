from datetime import datetime
import operator
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_pyarrow_numpy_string_invalid(self):
    pytest.importorskip('pyarrow')
    ser = Series([False, True])
    ser2 = Series(['a', 'b'], dtype='string[pyarrow_numpy]')
    result = ser == ser2
    expected = Series(False, index=ser.index)
    tm.assert_series_equal(result, expected)
    result = ser != ser2
    expected = Series(True, index=ser.index)
    tm.assert_series_equal(result, expected)
    with pytest.raises(TypeError, match='Invalid comparison'):
        ser > ser2