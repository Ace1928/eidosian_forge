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
def test_to_string_left_justify_cols(self):
    df = DataFrame({'x': [3234, 0.253]})
    df_s = df.to_string(justify='left')
    expected = '   x       \n0  3234.000\n1     0.253'
    assert df_s == expected