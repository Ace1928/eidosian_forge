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
@pytest.mark.parametrize('value,expected', [([9.4444], '   0\n0  9'), ([0.49], '       0\n0  5e-01'), ([10.9999], '    0\n0  11'), ([9.5444, 9.6], '    0\n0  10\n1  10'), ([0.46, 0.78, -9.9999], '       0\n0  5e-01\n1  8e-01\n2 -1e+01')])
def test_set_option_precision(self, value, expected):
    with option_context('display.precision', 0):
        df_value = DataFrame(value)
        assert str(df_value) == expected