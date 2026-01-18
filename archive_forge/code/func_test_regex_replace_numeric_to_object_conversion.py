from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_regex_replace_numeric_to_object_conversion(self, mix_abc):
    df = DataFrame(mix_abc)
    expec = DataFrame({'a': ['a', 1, 2, 3], 'b': mix_abc['b'], 'c': mix_abc['c']})
    res = df.replace(0, 'a')
    tm.assert_frame_equal(res, expec)
    assert res.a.dtype == np.object_