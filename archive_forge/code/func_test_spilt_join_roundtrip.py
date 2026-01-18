from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.strings.accessor import StringMethods
from pandas.tests.strings import object_pyarrow_numpy
def test_spilt_join_roundtrip(any_string_dtype):
    ser = Series(['a_b_c', 'c_d_e', np.nan, 'f_g_h'], dtype=any_string_dtype)
    result = ser.str.split('_').str.join('_')
    expected = ser.astype(object)
    tm.assert_series_equal(result, expected)