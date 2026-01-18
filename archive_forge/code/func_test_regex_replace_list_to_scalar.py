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
def test_regex_replace_list_to_scalar(self, mix_abc):
    df = DataFrame(mix_abc)
    expec = DataFrame({'a': mix_abc['a'], 'b': np.array([np.nan] * 4), 'c': [np.nan, np.nan, np.nan, 'd']})
    msg = 'Downcasting behavior in `replace`'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res = df.replace(['\\s*\\.\\s*', 'a|b'], np.nan, regex=True)
    res2 = df.copy()
    res3 = df.copy()
    with tm.assert_produces_warning(FutureWarning, match=msg):
        return_value = res2.replace(['\\s*\\.\\s*', 'a|b'], np.nan, regex=True, inplace=True)
    assert return_value is None
    with tm.assert_produces_warning(FutureWarning, match=msg):
        return_value = res3.replace(regex=['\\s*\\.\\s*', 'a|b'], value=np.nan, inplace=True)
    assert return_value is None
    tm.assert_frame_equal(res, expec)
    tm.assert_frame_equal(res2, expec)
    tm.assert_frame_equal(res3, expec)