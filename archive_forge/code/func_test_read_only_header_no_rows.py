from collections import namedtuple
from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm
@skip_pyarrow
@pytest.mark.parametrize('kwargs', [{}, {'index_col': False}])
def test_read_only_header_no_rows(all_parsers, kwargs):
    parser = all_parsers
    expected = DataFrame(columns=['a', 'b', 'c'])
    result = parser.read_csv(StringIO('a,b,c'), **kwargs)
    tm.assert_frame_equal(result, expected)