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
@pytest.mark.parametrize('replacer', [Timestamp('20170827'), np.int8(1), np.int16(1), np.float32(1), np.float64(1)])
def test_replace_replacer_dtype(self, replacer):
    df = DataFrame(['a'])
    msg = 'Downcasting behavior in `replace` '
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = df.replace({'a': replacer, 'b': replacer})
    expected = DataFrame([replacer])
    tm.assert_frame_equal(result, expected)