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
def test_to_string_header(self):
    ser = Series(range(10), dtype='int64')
    ser.index.name = 'foo'
    res = ser.to_string(header=True, max_rows=2)
    exp = 'foo\n0    0\n    ..\n9    9'
    assert res == exp
    res = ser.to_string(header=False, max_rows=2)
    exp = '0    0\n    ..\n9    9'
    assert res == exp