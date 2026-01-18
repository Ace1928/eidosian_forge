from datetime import datetime
from io import StringIO
import numpy as np
import pytest
from pandas.errors import EmptyDataError
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('data,exp_data', [('id,text,num_lines\n1,"line \n\'11\' line 12",2\n2,"line \n\'21\' line 22",2\n3,"line \n\'31\' line 32",1', [[2, "line \n'21' line 22", 2], [3, "line \n'31' line 32", 1]]), ('id,text,num_lines\n1,"line \'11\n\' line 12",2\n2,"line \'21\n\' line 22",2\n3,"line \'31\n\' line 32",1', [[2, "line '21\n' line 22", 2], [3, "line '31\n' line 32", 1]]), ('id,text,num_lines\n1,"line \'11\n\' \r\tline 12",2\n2,"line \'21\n\' \r\tline 22",2\n3,"line \'31\n\' \r\tline 32",1', [[2, "line '21\n' \r\tline 22", 2], [3, "line '31\n' \r\tline 32", 1]])])
@xfail_pyarrow
def test_skip_row_with_newline_and_quote(all_parsers, data, exp_data):
    parser = all_parsers
    result = parser.read_csv(StringIO(data), skiprows=[1])
    expected = DataFrame(exp_data, columns=['id', 'text', 'num_lines'])
    tm.assert_frame_equal(result, expected)