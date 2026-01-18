from datetime import datetime
from io import StringIO
import numpy as np
import pytest
from pandas.errors import EmptyDataError
from pandas import (
import pandas._testing as tm
@xfail_pyarrow
def test_skip_row_with_quote(all_parsers):
    parser = all_parsers
    data = 'id,text,num_lines\n1,"line \'11\' line 12",2\n2,"line \'21\' line 22",2\n3,"line \'31\' line 32",1'
    exp_data = [[2, "line '21' line 22", 2], [3, "line '31' line 32", 1]]
    expected = DataFrame(exp_data, columns=['id', 'text', 'num_lines'])
    result = parser.read_csv(StringIO(data), skiprows=[1])
    tm.assert_frame_equal(result, expected)