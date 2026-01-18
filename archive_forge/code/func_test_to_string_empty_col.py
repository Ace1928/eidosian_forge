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
def test_to_string_empty_col(self):
    ser = Series(['', 'Hello', 'World', '', '', 'Mooooo', '', ''])
    res = ser.to_string(index=False)
    exp = '      \n Hello\n World\n      \n      \nMooooo\n      \n      '
    assert re.match(exp, res)