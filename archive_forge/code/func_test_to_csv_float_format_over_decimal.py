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
def test_to_csv_float_format_over_decimal(self):
    df = DataFrame({'a': [0.5, 1.0]})
    result = df.to_csv(decimal=',', float_format=lambda x: np.format_float_positional(x, trim='-'), index=False)
    expected_rows = ['a', '0.5', '1']
    expected = tm.convert_rows_list_to_csv_str(expected_rows)
    assert result == expected