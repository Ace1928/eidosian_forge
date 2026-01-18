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
def test_to_csv_with_dst_transitions_with_pickle(self):
    idx = date_range('2015-01-01', '2015-12-31', freq='h', tz='Europe/Paris')
    idx = idx._with_freq(None)
    idx._data._freq = None
    df = DataFrame({'values': 1, 'idx': idx}, index=idx)
    with tm.ensure_clean('csv_date_format_with_dst') as path:
        df.to_csv(path, index=True)
        result = read_csv(path, index_col=0)
        result.index = to_datetime(result.index, utc=True).tz_convert('Europe/Paris')
        result['idx'] = to_datetime(result['idx'], utc=True).astype('datetime64[ns, Europe/Paris]')
        tm.assert_frame_equal(result, df)
    df.astype(str)
    with tm.ensure_clean('csv_date_format_with_dst') as path:
        df.to_pickle(path)
        result = pd.read_pickle(path)
        tm.assert_frame_equal(result, df)