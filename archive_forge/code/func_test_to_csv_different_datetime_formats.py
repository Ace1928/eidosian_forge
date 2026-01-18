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
def test_to_csv_different_datetime_formats(self):
    df = DataFrame({'date': pd.to_datetime('1970-01-01'), 'datetime': pd.date_range('1970-01-01', periods=2, freq='h')})
    expected_rows = ['date,datetime', '1970-01-01,1970-01-01 00:00:00', '1970-01-01,1970-01-01 01:00:00']
    expected = tm.convert_rows_list_to_csv_str(expected_rows)
    assert df.to_csv(index=False) == expected