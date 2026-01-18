from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_regex_replace_series_of_regexes(self, mix_abc):
    df = DataFrame(mix_abc)
    s1 = Series({'b': '\\s*\\.\\s*'})
    s2 = Series({'b': np.nan})
    res = df.replace(s1, s2, regex=True)
    res2 = df.copy()
    return_value = res2.replace(s1, s2, inplace=True, regex=True)
    assert return_value is None
    res3 = df.copy()
    return_value = res3.replace(regex=s1, value=s2, inplace=True)
    assert return_value is None
    expec = DataFrame({'a': mix_abc['a'], 'b': ['a', 'b', np.nan, np.nan], 'c': mix_abc['c']})
    tm.assert_frame_equal(res, expec)
    tm.assert_frame_equal(res2, expec)
    tm.assert_frame_equal(res3, expec)