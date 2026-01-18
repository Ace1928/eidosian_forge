from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_replace_bool_with_string(self):
    df = DataFrame({'a': [True, False], 'b': list('ab')})
    result = df.replace(True, 'a')
    expected = DataFrame({'a': ['a', False], 'b': df.b})
    tm.assert_frame_equal(result, expected)