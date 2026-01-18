from collections import defaultdict
from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_nullable_int_dtype(all_parsers, any_int_ea_dtype):
    parser = all_parsers
    dtype = any_int_ea_dtype
    data = 'a,b,c\n,3,5\n1,,6\n2,4,'
    expected = DataFrame({'a': pd.array([pd.NA, 1, 2], dtype=dtype), 'b': pd.array([3, pd.NA, 4], dtype=dtype), 'c': pd.array([5, 6, pd.NA], dtype=dtype)})
    actual = parser.read_csv(StringIO(data), dtype=dtype)
    tm.assert_frame_equal(actual, expected)