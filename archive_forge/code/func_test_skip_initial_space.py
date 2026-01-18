from datetime import datetime
from inspect import signature
from io import StringIO
import os
from pathlib import Path
import sys
import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
from pandas.io.parsers import TextFileReader
from pandas.io.parsers.c_parser_wrapper import CParserWrapper
def test_skip_initial_space(all_parsers):
    data = '"09-Apr-2012", "01:10:18.300", 2456026.548822908, 12849, 1.00361,  1.12551, 330.65659, 0355626618.16711,  73.48821, 314.11625,  1917.09447,   179.71425,  80.000, 240.000, -350,  70.06056, 344.98370, 1,   1, -0.689265, -0.692787,  0.212036,    14.7674,   41.605,   -9999.0,   -9999.0,   -9999.0,   -9999.0,   -9999.0,  -9999.0, 000, 012, 128'
    parser = all_parsers
    if parser.engine == 'pyarrow':
        msg = "The 'skipinitialspace' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), names=list(range(33)), header=None, na_values=['-9999.0'], skipinitialspace=True)
        return
    result = parser.read_csv(StringIO(data), names=list(range(33)), header=None, na_values=['-9999.0'], skipinitialspace=True)
    expected = DataFrame([['09-Apr-2012', '01:10:18.300', 2456026.548822908, 12849, 1.00361, 1.12551, 330.65659, 355626618.16711, 73.48821, 314.11625, 1917.09447, 179.71425, 80.0, 240.0, -350, 70.06056, 344.9837, 1, 1, -0.689265, -0.692787, 0.212036, 14.7674, 41.605, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0, 12, 128]])
    tm.assert_frame_equal(result, expected)