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
def test_to_csv_float_ea_float_format(self):
    df = DataFrame({'a': [1.1, 2.02, pd.NA, 6.000006], 'b': 'c'})
    df['a'] = df['a'].astype('Float64')
    result = df.to_csv(index=False, float_format='%.5f')
    expected = tm.convert_rows_list_to_csv_str(['a,b', '1.10000,c', '2.02000,c', ',c', '6.00001,c'])
    assert result == expected