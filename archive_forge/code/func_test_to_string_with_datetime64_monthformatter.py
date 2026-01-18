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
def test_to_string_with_datetime64_monthformatter(self):
    months = [datetime(2016, 1, 1), datetime(2016, 2, 2)]
    x = DataFrame({'months': months})

    def format_func(x):
        return x.strftime('%Y-%m')
    result = x.to_string(formatters={'months': format_func})
    expected = dedent('            months\n            0 2016-01\n            1 2016-02')
    assert result.strip() == expected