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
def test_to_string_name(self):
    ser = Series(range(100), dtype='int64')
    ser.name = 'myser'
    res = ser.to_string(max_rows=2, name=True)
    exp = '0      0\n      ..\n99    99\nName: myser'
    assert res == exp
    res = ser.to_string(max_rows=2, name=False)
    exp = '0      0\n      ..\n99    99'
    assert res == exp