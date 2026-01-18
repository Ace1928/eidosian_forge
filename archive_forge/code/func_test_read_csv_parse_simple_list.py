import csv
from io import StringIO
import pytest
from pandas import DataFrame
import pandas._testing as tm
from pandas.io.parsers import TextParser
def test_read_csv_parse_simple_list(all_parsers):
    parser = all_parsers
    data = 'foo\nbar baz\nqux foo\nfoo\nbar'
    result = parser.read_csv(StringIO(data), header=None)
    expected = DataFrame(['foo', 'bar baz', 'qux foo', 'foo', 'bar'])
    tm.assert_frame_equal(result, expected)