from io import StringIO
import numpy as np
import pytest
from pandas._libs.parsers import STR_NA_VALUES
from pandas import (
import pandas._testing as tm
def test_string_nas(all_parsers):
    parser = all_parsers
    data = 'A,B,C\na,b,c\nd,,f\n,g,h\n'
    result = parser.read_csv(StringIO(data))
    expected = DataFrame([['a', 'b', 'c'], ['d', np.nan, 'f'], [np.nan, 'g', 'h']], columns=['A', 'B', 'C'])
    if parser.engine == 'pyarrow':
        expected.loc[2, 'A'] = None
        expected.loc[1, 'B'] = None
    tm.assert_frame_equal(result, expected)