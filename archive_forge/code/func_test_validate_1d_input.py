from collections import defaultdict
from datetime import datetime
from functools import partial
import math
import operator
import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.api import (
@pytest.mark.parametrize('dtype', [None, np.int64, np.uint64, np.float64])
def test_validate_1d_input(dtype):
    msg = 'Index data must be 1-dimensional'
    arr = np.arange(8).reshape(2, 2, 2)
    with pytest.raises(ValueError, match=msg):
        Index(arr, dtype=dtype)
    df = DataFrame(arr.reshape(4, 2))
    with pytest.raises(ValueError, match=msg):
        Index(df, dtype=dtype)
    ser = Series(0, range(4))
    with pytest.raises(ValueError, match=msg):
        ser.index = np.array([[2, 3]] * 4, dtype=dtype)