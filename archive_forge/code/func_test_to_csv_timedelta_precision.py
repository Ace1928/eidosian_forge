import io
import os
import sys
from zipfile import ZipFile
from _csv import Error
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_to_csv_timedelta_precision(self):
    s = pd.Series([1, 1]).astype('timedelta64[ns]')
    buf = io.StringIO()
    s.to_csv(buf)
    result = buf.getvalue()
    expected_rows = [',0', '0,0 days 00:00:00.000000001', '1,0 days 00:00:00.000000001']
    expected = tm.convert_rows_list_to_csv_str(expected_rows)
    assert result == expected