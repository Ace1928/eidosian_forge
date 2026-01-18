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
@pytest.mark.parametrize('nrows', [10, 98, 99, 100, 101, 102])
def test_to_csv_dup_cols(self, nrows):
    df = DataFrame(np.ones((nrows, 3)), index=Index([f'i-{i}' for i in range(nrows)], name='a'), columns=Index([f'i-{i}' for i in range(3)], name='a'))
    cols = list(df.columns)
    cols[:2] = ['dupe', 'dupe']
    cols[-2:] = ['dupe', 'dupe']
    ix = list(df.index)
    ix[:2] = ['rdupe', 'rdupe']
    ix[-2:] = ['rdupe', 'rdupe']
    df.index = ix
    df.columns = cols
    result, expected = self._return_result_expected(df, 1000, dupe_col=True)
    tm.assert_frame_equal(result, expected, check_names=False)