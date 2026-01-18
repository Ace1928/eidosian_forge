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
def test_skiprows_inference():
    data = '\nText contained in the file header\n\nDataCol1   DataCol2\n     0.0        1.0\n   101.6      956.1\n'.strip()
    skiprows = 2
    depr_msg = "The 'delim_whitespace' keyword in pd.read_csv is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        expected = read_csv(StringIO(data), skiprows=skiprows, delim_whitespace=True)
    result = read_fwf(StringIO(data), skiprows=skiprows)
    tm.assert_frame_equal(result, expected)