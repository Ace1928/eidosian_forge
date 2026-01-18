from datetime import datetime
from io import (
from pathlib import Path
import numpy as np
import pytest
from pandas.errors import EmptyDataError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.io.common import urlopen
from pandas.io.parsers import (
def test_skiprows_by_index_inference():
    data = '\nTo be skipped\nNot  To  Be  Skipped\nOnce more to be skipped\n123  34   8      123\n456  78   9      456\n'.strip()
    skiprows = [0, 2]
    depr_msg = "The 'delim_whitespace' keyword in pd.read_csv is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        expected = read_csv(StringIO(data), skiprows=skiprows, delim_whitespace=True)
    result = read_fwf(StringIO(data), skiprows=skiprows)
    tm.assert_frame_equal(result, expected)