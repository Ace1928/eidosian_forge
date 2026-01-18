from __future__ import annotations
from datetime import datetime
import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_replace_list(self):
    obj = {'a': list('ab..'), 'b': list('efgh'), 'c': list('helo')}
    dfobj = DataFrame(obj)
    to_replace_res = ['.', 'e']
    values = [np.nan, 'crap']
    res = dfobj.replace(to_replace_res, values)
    expec = DataFrame({'a': ['a', 'b', np.nan, np.nan], 'b': ['crap', 'f', 'g', 'h'], 'c': ['h', 'crap', 'l', 'o']})
    tm.assert_frame_equal(res, expec)
    to_replace_res = ['.', 'f']
    values = ['..', 'crap']
    res = dfobj.replace(to_replace_res, values)
    expec = DataFrame({'a': ['a', 'b', '..', '..'], 'b': ['e', 'crap', 'g', 'h'], 'c': ['h', 'e', 'l', 'o']})
    tm.assert_frame_equal(res, expec)