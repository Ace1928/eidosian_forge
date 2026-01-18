from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('metachar', ['[]', '()', '\\d', '\\w', '\\s'])
def test_replace_regex_metachar(self, metachar):
    df = DataFrame({'a': [metachar, 'else']})
    result = df.replace({'a': {metachar: 'paren'}})
    expected = DataFrame({'a': ['paren', 'else']})
    tm.assert_frame_equal(result, expected)