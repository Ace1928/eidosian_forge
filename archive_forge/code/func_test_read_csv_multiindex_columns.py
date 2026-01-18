from collections import namedtuple
from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm
@xfail_pyarrow
def test_read_csv_multiindex_columns(all_parsers):
    parser = all_parsers
    s1 = 'Male, Male, Male, Female, Female\nR, R, L, R, R\n.86, .67, .88, .78, .81'
    s2 = 'Male, Male, Male, Female, Female\nR, R, L, R, R\n.86, .67, .88, .78, .81\n.86, .67, .88, .78, .82'
    mi = MultiIndex.from_tuples([('Male', 'R'), (' Male', ' R'), (' Male', ' L'), (' Female', ' R'), (' Female', ' R.1')])
    expected = DataFrame([[0.86, 0.67, 0.88, 0.78, 0.81], [0.86, 0.67, 0.88, 0.78, 0.82]], columns=mi)
    df1 = parser.read_csv(StringIO(s1), header=[0, 1])
    tm.assert_frame_equal(df1, expected.iloc[:1])
    df2 = parser.read_csv(StringIO(s2), header=[0, 1])
    tm.assert_frame_equal(df2, expected)