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
def test_to_string_float_format(self):
    ser = Series(range(10), dtype='float64')
    res = ser.to_string(float_format=lambda x: f'{x:2.1f}', max_rows=2)
    exp = '0   0.0\n     ..\n9   9.0'
    assert res == exp