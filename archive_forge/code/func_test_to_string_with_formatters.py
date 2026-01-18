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
def test_to_string_with_formatters(self):
    df = DataFrame({'int': [1, 2, 3], 'float': [1.0, 2.0, 3.0], 'object': [(1, 2), True, False]}, columns=['int', 'float', 'object'])
    formatters = [('int', lambda x: f'0x{x:x}'), ('float', lambda x: f'[{x: 4.1f}]'), ('object', lambda x: f'-{x!s}-')]
    result = df.to_string(formatters=dict(formatters))
    result2 = df.to_string(formatters=list(zip(*formatters))[1])
    assert result == '  int  float    object\n0 0x1 [ 1.0]  -(1, 2)-\n1 0x2 [ 2.0]    -True-\n2 0x3 [ 3.0]   -False-'
    assert result == result2