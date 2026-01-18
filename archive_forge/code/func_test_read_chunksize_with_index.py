from io import StringIO
import numpy as np
import pytest
from pandas._libs import parsers as libparsers
from pandas.errors import DtypeWarning
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('index_col', [0, 'index'])
def test_read_chunksize_with_index(all_parsers, index_col):
    parser = all_parsers
    data = 'index,A,B,C,D\nfoo,2,3,4,5\nbar,7,8,9,10\nbaz,12,13,14,15\nqux,12,13,14,15\nfoo2,12,13,14,15\nbar2,12,13,14,15\n'
    expected = DataFrame([['foo', 2, 3, 4, 5], ['bar', 7, 8, 9, 10], ['baz', 12, 13, 14, 15], ['qux', 12, 13, 14, 15], ['foo2', 12, 13, 14, 15], ['bar2', 12, 13, 14, 15]], columns=['index', 'A', 'B', 'C', 'D'])
    expected = expected.set_index('index')
    if parser.engine == 'pyarrow':
        msg = "The 'chunksize' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            with parser.read_csv(StringIO(data), index_col=0, chunksize=2) as reader:
                list(reader)
        return
    with parser.read_csv(StringIO(data), index_col=0, chunksize=2) as reader:
        chunks = list(reader)
    tm.assert_frame_equal(chunks[0], expected[:2])
    tm.assert_frame_equal(chunks[1], expected[2:4])
    tm.assert_frame_equal(chunks[2], expected[4:])