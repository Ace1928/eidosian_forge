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
def test_to_csv_date_format_in_categorical(self):
    ser = pd.Series(pd.to_datetime(['2021-03-27', pd.NaT], format='%Y-%m-%d'))
    ser = ser.astype('category')
    expected = tm.convert_rows_list_to_csv_str(['0', '2021-03-27', '""'])
    assert ser.to_csv(index=False) == expected
    ser = pd.Series(pd.date_range(start='2021-03-27', freq='D', periods=1, tz='Europe/Berlin').append(pd.DatetimeIndex([pd.NaT])))
    ser = ser.astype('category')
    assert ser.to_csv(index=False, date_format='%Y-%m-%d') == expected