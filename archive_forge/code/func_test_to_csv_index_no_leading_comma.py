import csv
from io import StringIO
import os
import numpy as np
import pytest
from pandas.errors import ParserError
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
from pandas.io.common import get_handle
def test_to_csv_index_no_leading_comma(self):
    df = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}, index=['one', 'two', 'three'])
    buf = StringIO()
    df.to_csv(buf, index_label=False)
    expected_rows = ['A,B', 'one,1,4', 'two,2,5', 'three,3,6']
    expected = tm.convert_rows_list_to_csv_str(expected_rows)
    assert buf.getvalue() == expected