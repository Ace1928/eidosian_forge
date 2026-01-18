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
def test_replace_input_formats_scalar(self):
    df = DataFrame({'A': [np.nan, 0, np.inf], 'B': [0, 2, 5], 'C': ['', 'asdf', 'fd']})
    to_rep = {'A': np.nan, 'B': 0, 'C': ''}
    filled = df.replace(to_rep, 0)
    expected = {k: v.replace(to_rep[k], 0) for k, v in df.items()}
    tm.assert_frame_equal(filled, DataFrame(expected))
    msg = 'value argument must be scalar, dict, or Series'
    with pytest.raises(TypeError, match=msg):
        df.replace(to_rep, [np.nan, 0, ''])
    to_rep = [np.nan, 0, '']
    result = df.replace(to_rep, -1)
    expected = df.copy()
    for rep in to_rep:
        return_value = expected.replace(rep, -1, inplace=True)
        assert return_value is None
    tm.assert_frame_equal(result, expected)