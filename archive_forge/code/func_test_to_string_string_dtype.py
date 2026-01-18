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
def test_to_string_string_dtype(self):
    pytest.importorskip('pyarrow')
    df = DataFrame({'x': ['foo', 'bar', 'baz'], 'y': ['a', 'b', 'c'], 'z': [1, 2, 3]})
    df = df.astype({'x': 'string[pyarrow]', 'y': 'string[python]', 'z': 'int64[pyarrow]'})
    result = df.dtypes.to_string()
    expected = dedent('            x    string[pyarrow]\n            y     string[python]\n            z     int64[pyarrow]')
    assert result == expected