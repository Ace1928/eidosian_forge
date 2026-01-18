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
def test_to_csv_categorical_and_ea(self):
    df = DataFrame({'a': 'x', 'b': [1, pd.NA]})
    df['b'] = df['b'].astype('Int16')
    df['b'] = df['b'].astype('category')
    result = df.to_csv()
    expected_rows = [',a,b', '0,x,1', '1,x,']
    expected = tm.convert_rows_list_to_csv_str(expected_rows)
    assert result == expected