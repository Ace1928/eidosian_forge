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
def test_to_string_without_index(self):
    ser = Series([1, 2, 3, 4])
    result = ser.to_string(index=False)
    expected = '\n'.join(['1', '2', '3', '4'])
    assert result == expected