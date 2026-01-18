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
@pytest.mark.parametrize('nrows', [10, 98, 99, 100, 101, 102, 198, 199, 200, 201, 202, 249, 250, 251])
@pytest.mark.parametrize('ncols', [1, 2, 3, 4])
def test_to_csv_idx_ncols(self, nrows, ncols):
    df = DataFrame(np.ones((nrows, ncols)), index=Index([f'i-{i}' for i in range(nrows)], name='a'), columns=Index([f'i-{i}' for i in range(ncols)], name='a'))
    result, expected = self._return_result_expected(df, 1000)
    tm.assert_frame_equal(result, expected, check_names=False)