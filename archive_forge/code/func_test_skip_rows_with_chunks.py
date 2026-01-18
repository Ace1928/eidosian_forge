from datetime import datetime
from io import StringIO
import numpy as np
import pytest
from pandas.errors import EmptyDataError
from pandas import (
import pandas._testing as tm
@xfail_pyarrow
def test_skip_rows_with_chunks(all_parsers):
    data = 'col_a\n10\n20\n30\n40\n50\n60\n70\n80\n90\n100\n'
    parser = all_parsers
    reader = parser.read_csv(StringIO(data), engine=parser, skiprows=lambda x: x in [1, 4, 5], chunksize=4)
    df1 = next(reader)
    df2 = next(reader)
    tm.assert_frame_equal(df1, DataFrame({'col_a': [20, 30, 60, 70]}))
    tm.assert_frame_equal(df2, DataFrame({'col_a': [80, 90, 100]}, index=[4, 5, 6]))