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
@pytest.mark.parametrize('r_idx_type, c_idx_type', [('i', 'i'), ('s', 's'), ('s', 'dt'), ('p', 'p')])
@pytest.mark.parametrize('ncols', [1, 2, 3, 4])
@pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
def test_to_csv_idx_types(self, nrows, r_idx_type, c_idx_type, ncols):
    axes = {'i': lambda n: Index(np.arange(n), dtype=np.int64), 's': lambda n: Index([f'{i}_{chr(i)}' for i in range(97, 97 + n)]), 'dt': lambda n: date_range('2020-01-01', periods=n), 'p': lambda n: period_range('2020-01-01', periods=n, freq='D')}
    df = DataFrame(np.ones((nrows, ncols)), index=axes[r_idx_type](nrows), columns=axes[c_idx_type](ncols))
    result, expected = self._return_result_expected(df, 1000, r_idx_type, c_idx_type)
    tm.assert_frame_equal(result, expected, check_names=False)