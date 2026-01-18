from datetime import datetime
from io import StringIO
import numpy as np
import pytest
from pandas.errors import EmptyDataError
from pandas import (
import pandas._testing as tm
@xfail_pyarrow
@pytest.mark.parametrize('skiprows', [list(range(6)), 6])
def test_skip_rows_bug(all_parsers, skiprows):
    parser = all_parsers
    text = '#foo,a,b,c\n#foo,a,b,c\n#foo,a,b,c\n#foo,a,b,c\n#foo,a,b,c\n#foo,a,b,c\n1/1/2000,1.,2.,3.\n1/2/2000,4,5,6\n1/3/2000,7,8,9\n'
    result = parser.read_csv(StringIO(text), skiprows=skiprows, header=None, index_col=0, parse_dates=True)
    index = Index([datetime(2000, 1, 1), datetime(2000, 1, 2), datetime(2000, 1, 3)], name=0)
    expected = DataFrame(np.arange(1.0, 10.0).reshape((3, 3)), columns=[1, 2, 3], index=index)
    tm.assert_frame_equal(result, expected)