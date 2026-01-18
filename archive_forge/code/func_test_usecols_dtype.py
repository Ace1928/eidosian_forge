from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm
def test_usecols_dtype(all_parsers):
    parser = all_parsers
    data = '\ncol1,col2,col3\na,1,x\nb,2,y\n'
    result = parser.read_csv(StringIO(data), usecols=['col1', 'col2'], dtype={'col1': 'string', 'col2': 'uint8', 'col3': 'string'})
    expected = DataFrame({'col1': array(['a', 'b']), 'col2': np.array([1, 2], dtype='uint8')})
    tm.assert_frame_equal(result, expected)