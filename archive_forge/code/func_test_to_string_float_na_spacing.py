from datetime import (
from io import StringIO
import re
import sys
from textwrap import dedent
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
import pandas._testing as tm
def test_to_string_float_na_spacing(self):
    ser = Series([0.0, 1.5678, 2.0, -3.0, 4.0])
    ser[::2] = np.nan
    result = ser.to_string()
    expected = '0       NaN\n1    1.5678\n2       NaN\n3   -3.0000\n4       NaN'
    assert result == expected