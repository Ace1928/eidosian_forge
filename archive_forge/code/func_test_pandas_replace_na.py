import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
def test_pandas_replace_na(self):
    ser = pd.Series(['AA', 'BB', 'CC', 'DD', 'EE', '', pd.NA], dtype='string')
    regex_mapping = {'AA': 'CC', 'BB': 'CC', 'EE': 'CC', 'CC': 'CC-REPL'}
    result = ser.replace(regex_mapping, regex=True)
    exp = pd.Series(['CC', 'CC', 'CC-REPL', 'DD', 'CC', '', pd.NA], dtype='string')
    tm.assert_series_equal(result, exp)