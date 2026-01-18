from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.xfail(using_pyarrow_string_dtype(), reason="can't set float into string")
def test_replace_dict_no_regex(self):
    answer = Series({0: 'Strongly Agree', 1: 'Agree', 2: 'Neutral', 3: 'Disagree', 4: 'Strongly Disagree'})
    weights = {'Agree': 4, 'Disagree': 2, 'Neutral': 3, 'Strongly Agree': 5, 'Strongly Disagree': 1}
    expected = Series({0: 5, 1: 4, 2: 3, 3: 2, 4: 1})
    msg = 'Downcasting behavior in `replace` '
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = answer.replace(weights)
    tm.assert_series_equal(result, expected)