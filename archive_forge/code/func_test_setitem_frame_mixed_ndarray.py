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
def test_setitem_frame_mixed_ndarray(self, float_string_frame):
    f = float_string_frame.copy()
    piece = float_string_frame.loc[f.index[:2], ['A', 'B']]
    key = (f.index[slice(-2, None)], ['A', 'B'])
    f.loc[key] = piece.values
    tm.assert_almost_equal(f.loc[f.index[-2:], ['A', 'B']].values, piece.values)