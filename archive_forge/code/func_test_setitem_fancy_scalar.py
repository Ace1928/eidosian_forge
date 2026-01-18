from collections import namedtuple
from datetime import (
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas._libs import iNaT
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer
import pandas as pd
from pandas import (
import pandas._testing as tm
@td.skip_array_manager_invalid_test
def test_setitem_fancy_scalar(self, float_frame):
    f = float_frame
    expected = float_frame.copy()
    ix = f.loc
    for j, col in enumerate(f.columns):
        f[col]
        for idx in f.index[::5]:
            i = f.index.get_loc(idx)
            val = np.random.default_rng(2).standard_normal()
            expected.iloc[i, j] = val
            ix[idx, col] = val
            tm.assert_frame_equal(f, expected)