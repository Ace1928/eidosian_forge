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
def test_to_string_line_width_no_index_no_header(self):
    df = DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
    df_s = df.to_string(line_width=1, index=False, header=False)
    expected = '1  \\\n2   \n3   \n\n4  \n5  \n6  '
    assert df_s == expected
    df = DataFrame({'x': [11, 22, 33], 'y': [4, 5, 6]})
    df_s = df.to_string(line_width=1, index=False, header=False)
    expected = '11  \\\n22   \n33   \n\n4  \n5  \n6  '
    assert df_s == expected
    df = DataFrame({'x': [11, 22, -33], 'y': [4, 5, -6]})
    df_s = df.to_string(line_width=1, index=False, header=False)
    expected = ' 11  \\\n 22   \n-33   \n\n 4  \n 5  \n-6  '
    assert df_s == expected