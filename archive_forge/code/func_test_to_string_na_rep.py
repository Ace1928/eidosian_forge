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
def test_to_string_na_rep(self):
    ser = Series(index=range(100), dtype=np.float64)
    res = ser.to_string(na_rep='foo', max_rows=2)
    exp = '0    foo\n      ..\n99   foo'
    assert res == exp