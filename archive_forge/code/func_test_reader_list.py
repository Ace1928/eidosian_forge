import csv
from io import StringIO
import pytest
from pandas import DataFrame
import pandas._testing as tm
from pandas.io.parsers import TextParser
def test_reader_list(all_parsers):
    data = 'index,A,B,C,D\nfoo,2,3,4,5\nbar,7,8,9,10\nbaz,12,13,14,15\nqux,12,13,14,15\nfoo2,12,13,14,15\nbar2,12,13,14,15\n'
    parser = all_parsers
    kwargs = {'index_col': 0}
    lines = list(csv.reader(StringIO(data)))
    with TextParser(lines, chunksize=2, **kwargs) as reader:
        chunks = list(reader)
    expected = parser.read_csv(StringIO(data), **kwargs)
    tm.assert_frame_equal(chunks[0], expected[:2])
    tm.assert_frame_equal(chunks[1], expected[2:4])
    tm.assert_frame_equal(chunks[2], expected[4:])