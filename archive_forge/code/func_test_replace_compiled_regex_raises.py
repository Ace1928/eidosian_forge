from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
from pandas.tests.strings import (
def test_replace_compiled_regex_raises(any_string_dtype):
    ser = Series(['fooBAD__barBAD__bad', np.nan], dtype=any_string_dtype)
    pat = re.compile('BAD_*')
    msg = 'case and flags cannot be set when pat is a compiled regex'
    with pytest.raises(ValueError, match=msg):
        ser.str.replace(pat, '', flags=re.IGNORECASE, regex=True)
    with pytest.raises(ValueError, match=msg):
        ser.str.replace(pat, '', case=False, regex=True)
    with pytest.raises(ValueError, match=msg):
        ser.str.replace(pat, '', case=True, regex=True)