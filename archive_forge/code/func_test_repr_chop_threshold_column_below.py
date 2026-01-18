from datetime import datetime
from io import StringIO
from pathlib import Path
import re
from shutil import get_terminal_size
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
from pandas.io.formats import printing
import pandas.io.formats.format as fmt
def test_repr_chop_threshold_column_below(self):
    df = DataFrame([[10, 20, 30, 40], [8e-10, -1e-11, 2e-09, -2e-11]]).T
    with option_context('display.chop_threshold', 0):
        assert repr(df) == '      0             1\n0  10.0  8.000000e-10\n1  20.0 -1.000000e-11\n2  30.0  2.000000e-09\n3  40.0 -2.000000e-11'
    with option_context('display.chop_threshold', 1e-08):
        assert repr(df) == '      0             1\n0  10.0  0.000000e+00\n1  20.0  0.000000e+00\n2  30.0  0.000000e+00\n3  40.0  0.000000e+00'
    with option_context('display.chop_threshold', 5e-11):
        assert repr(df) == '      0             1\n0  10.0  8.000000e-10\n1  20.0  0.000000e+00\n2  30.0  2.000000e-09\n3  40.0  0.000000e+00'