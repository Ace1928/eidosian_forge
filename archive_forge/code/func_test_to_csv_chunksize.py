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
def test_to_csv_chunksize(self):
    chunksize = 1000
    rows = chunksize // 2 + 1
    df = DataFrame(np.ones((rows, 2)), columns=Index(list('ab'), dtype=object), index=MultiIndex.from_arrays([range(rows) for _ in range(2)]))
    result, expected = self._return_result_expected(df, chunksize, rnlvl=2)
    tm.assert_frame_equal(result, expected, check_names=False)