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
def test_nullable_float_to_string(self, float_ea_dtype):
    dtype = float_ea_dtype
    ser = Series([0.0, 1.0, None], dtype=dtype)
    result = ser.to_string()
    expected = dedent('            0     0.0\n            1     1.0\n            2    <NA>')
    assert result == expected