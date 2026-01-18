from datetime import (
from io import StringIO
import re
import sys
from textwrap import dedent
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
import pandas._testing as tm
def test_to_string_float_format_no_fixed_width(self):
    df = DataFrame({'x': [0.19999]})
    expected = '      x\n0 0.200'
    assert df.to_string(float_format='%.3f') == expected
    df = DataFrame({'x': [100.0]})
    expected = '    x\n0 100'
    assert df.to_string(float_format='%.0f') == expected