from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm
def test_usecols_index_col_conflict2(all_parsers):
    parser = all_parsers
    data = 'a,b,c,d\nA,a,1,one\nB,b,2,two'
    expected = DataFrame({'b': ['a', 'b'], 'c': [1, 2], 'd': ('one', 'two')})
    expected = expected.set_index(['b', 'c'])
    result = parser.read_csv(StringIO(data), usecols=['b', 'c', 'd'], index_col=['b', 'c'])
    tm.assert_frame_equal(result, expected)