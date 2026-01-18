from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm
def test_usecols_with_names(all_parsers):
    data = 'a,b,c\n1,2,3\n4,5,6\n7,8,9\n10,11,12'
    parser = all_parsers
    names = ['foo', 'bar']
    if parser.engine == 'pyarrow':
        with pytest.raises(ValueError, match=_msg_pyarrow_requires_names):
            parser.read_csv(StringIO(data), names=names, usecols=[1, 2], header=0)
        return
    result = parser.read_csv(StringIO(data), names=names, usecols=[1, 2], header=0)
    expected = DataFrame([[2, 3], [5, 6], [8, 9], [11, 12]], columns=names)
    tm.assert_frame_equal(result, expected)