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
def test_to_string_no_index(self):
    df = DataFrame({'x': [11, 22], 'y': [33, -44], 'z': ['AAA', '   ']})
    df_s = df.to_string(index=False)
    expected = ' x   y   z\n11  33 AAA\n22 -44    '
    assert df_s == expected
    df_s = df[['y', 'x', 'z']].to_string(index=False)
    expected = '  y  x   z\n 33 11 AAA\n-44 22    '
    assert df_s == expected