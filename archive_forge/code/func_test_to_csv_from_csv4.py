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
def test_to_csv_from_csv4(self):
    with tm.ensure_clean('__tmp_to_csv_from_csv4__') as path:
        dt = pd.Timedelta(seconds=1)
        df = DataFrame({'dt_data': [i * dt for i in range(3)]}, index=Index([i * dt for i in range(3)], name='dt_index'))
        df.to_csv(path)
        result = read_csv(path, index_col='dt_index')
        result.index = pd.to_timedelta(result.index)
        result['dt_data'] = pd.to_timedelta(result['dt_data'])
        tm.assert_frame_equal(df, result, check_index_type=True)