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
def test_to_csv_empty(self):
    df = DataFrame(index=np.arange(10, dtype=np.int64))
    result, expected = self._return_result_expected(df, 1000)
    tm.assert_frame_equal(result, expected, check_column_type=False)