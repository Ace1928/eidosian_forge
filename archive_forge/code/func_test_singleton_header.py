from collections import namedtuple
from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm
@xfail_pyarrow
def test_singleton_header(all_parsers):
    data = 'a,b,c\n0,1,2\n1,2,3'
    parser = all_parsers
    expected = DataFrame({'a': [0, 1], 'b': [1, 2], 'c': [2, 3]})
    result = parser.read_csv(StringIO(data), header=[0])
    tm.assert_frame_equal(result, expected)