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
@pytest.mark.parametrize('ncols', [2, 3, 4])
@pytest.mark.parametrize('df_params, func_params', [[{'r_idx_nlevels': 2}, {'rnlvl': 2}], [{'c_idx_nlevels': 2}, {'cnlvl': 2}], [{'r_idx_nlevels': 2, 'c_idx_nlevels': 2}, {'rnlvl': 2, 'cnlvl': 2}]])
def test_to_csv_params(self, nrows, df_params, func_params, ncols):
    if df_params.get('r_idx_nlevels'):
        index = MultiIndex.from_arrays(([f'i-{i}' for i in range(nrows)] for _ in range(df_params['r_idx_nlevels'])))
    else:
        index = None
    if df_params.get('c_idx_nlevels'):
        columns = MultiIndex.from_arrays(([f'i-{i}' for i in range(ncols)] for _ in range(df_params['c_idx_nlevels'])))
    else:
        columns = Index([f'i-{i}' for i in range(ncols)], dtype=object)
    df = DataFrame(np.ones((nrows, ncols)), index=index, columns=columns)
    result, expected = self._return_result_expected(df, 1000, **func_params)
    tm.assert_frame_equal(result, expected, check_names=False)