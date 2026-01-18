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
def test_replace_swapping_bug(self, using_infer_string):
    df = DataFrame({'a': [True, False, True]})
    res = df.replace({'a': {True: 'Y', False: 'N'}})
    expect = DataFrame({'a': ['Y', 'N', 'Y']})
    tm.assert_frame_equal(res, expect)
    df = DataFrame({'a': [0, 1, 0]})
    res = df.replace({'a': {0: 'Y', 1: 'N'}})
    expect = DataFrame({'a': ['Y', 'N', 'Y']})
    tm.assert_frame_equal(res, expect)