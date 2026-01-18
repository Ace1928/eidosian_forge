from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
from pandas.tests.strings import (
def test_replace_literal_callable_raises(any_string_dtype):
    ser = Series([], dtype=any_string_dtype)
    repl = lambda m: m.group(0).swapcase()
    msg = 'Cannot use a callable replacement when regex=False'
    with pytest.raises(ValueError, match=msg):
        ser.str.replace('abc', repl, regex=False)