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
def test_to_string_masked_ea_with_formatter(self):
    df = DataFrame({'a': Series([0.123456789, 1.123456789], dtype='Float64'), 'b': Series([1, 2], dtype='Int64')})
    result = df.to_string(formatters=['{:.2f}'.format, '{:.2f}'.format])
    expected = dedent('                  a     b\n            0  0.12  1.00\n            1  1.12  2.00')
    assert result == expected