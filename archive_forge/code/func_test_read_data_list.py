import csv
from io import StringIO
import pytest
from pandas import DataFrame
import pandas._testing as tm
from pandas.io.parsers import TextParser
@xfail_pyarrow
def test_read_data_list(all_parsers):
    parser = all_parsers
    kwargs = {'index_col': 0}
    data = 'A,B,C\nfoo,1,2,3\nbar,4,5,6'
    data_list = [['A', 'B', 'C'], ['foo', '1', '2', '3'], ['bar', '4', '5', '6']]
    expected = parser.read_csv(StringIO(data), **kwargs)
    with TextParser(data_list, chunksize=2, **kwargs) as parser:
        result = parser.read()
    tm.assert_frame_equal(result, expected)