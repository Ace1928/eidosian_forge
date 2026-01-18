import csv
from io import StringIO
import pytest
from pandas.compat import PY311
from pandas.errors import ParserError
from pandas import DataFrame
import pandas._testing as tm
def test_quote_char_basic(all_parsers):
    parser = all_parsers
    data = 'a,b,c\n1,2,"cat"'
    expected = DataFrame([[1, 2, 'cat']], columns=['a', 'b', 'c'])
    result = parser.read_csv(StringIO(data), quotechar='"')
    tm.assert_frame_equal(result, expected)