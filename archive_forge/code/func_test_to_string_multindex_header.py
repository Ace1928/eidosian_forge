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
def test_to_string_multindex_header(self):
    df = DataFrame({'a': [0], 'b': [1], 'c': [2], 'd': [3]}).set_index(['a', 'b'])
    res = df.to_string(header=['r1', 'r2'])
    exp = '    r1 r2\na b      \n0 1  2  3'
    assert res == exp