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
@pytest.mark.slow
@pytest.mark.parametrize('nrows', [2, 10, 99, 100, 101, 102, 198, 199, 200, 201, 202, 249, 250, 251])
def test_to_csv_nrows(self, nrows):
    df = DataFrame(np.ones((nrows, 4)), index=date_range('2020-01-01', periods=nrows), columns=Index(list('abcd'), dtype=object))
    result, expected = self._return_result_expected(df, 1000, 'dt', 's')
    tm.assert_frame_equal(result, expected, check_names=False)