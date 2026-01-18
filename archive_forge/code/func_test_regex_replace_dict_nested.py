from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_regex_replace_dict_nested(self, mix_abc):
    dfmix = DataFrame(mix_abc)
    res = dfmix.replace({'b': {'\\s*\\.\\s*': np.nan}}, regex=True)
    res2 = dfmix.copy()
    res4 = dfmix.copy()
    return_value = res2.replace({'b': {'\\s*\\.\\s*': np.nan}}, inplace=True, regex=True)
    assert return_value is None
    res3 = dfmix.replace(regex={'b': {'\\s*\\.\\s*': np.nan}})
    return_value = res4.replace(regex={'b': {'\\s*\\.\\s*': np.nan}}, inplace=True)
    assert return_value is None
    expec = DataFrame({'a': mix_abc['a'], 'b': ['a', 'b', np.nan, np.nan], 'c': mix_abc['c']})
    tm.assert_frame_equal(res, expec)
    tm.assert_frame_equal(res2, expec)
    tm.assert_frame_equal(res3, expec)
    tm.assert_frame_equal(res4, expec)