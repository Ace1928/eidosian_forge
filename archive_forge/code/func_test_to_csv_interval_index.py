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
def test_to_csv_interval_index(self, using_infer_string):
    df = DataFrame({'A': list('abc'), 'B': range(3)}, index=pd.interval_range(0, 3))
    with tm.ensure_clean('__tmp_to_csv_interval_index__.csv') as path:
        df.to_csv(path)
        result = self.read_csv(path, index_col=0)
        expected = df.copy()
        if using_infer_string:
            expected.index = expected.index.astype('string[pyarrow_numpy]')
        else:
            expected.index = expected.index.astype(str)
        tm.assert_frame_equal(result, expected)