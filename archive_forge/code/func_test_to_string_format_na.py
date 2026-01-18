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
def test_to_string_format_na(self):
    df = DataFrame({'A': [np.nan, -1, -2.1234, 3, 4], 'B': [np.nan, 'foo', 'foooo', 'fooooo', 'bar']})
    result = df.to_string()
    expected = '        A       B\n0     NaN     NaN\n1 -1.0000     foo\n2 -2.1234   foooo\n3  3.0000  fooooo\n4  4.0000     bar'
    assert result == expected
    df = DataFrame({'A': [np.nan, -1.0, -2.0, 3.0, 4.0], 'B': [np.nan, 'foo', 'foooo', 'fooooo', 'bar']})
    result = df.to_string()
    expected = '     A       B\n0  NaN     NaN\n1 -1.0     foo\n2 -2.0   foooo\n3  3.0  fooooo\n4  4.0     bar'
    assert result == expected