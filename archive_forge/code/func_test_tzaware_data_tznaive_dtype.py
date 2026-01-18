import array
from collections import (
from collections.abc import Iterator
from dataclasses import make_dataclass
from datetime import (
import functools
import re
import numpy as np
from numpy import ma
from numpy.ma import mrecords
import pytest
import pytz
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.arrays import (
def test_tzaware_data_tznaive_dtype(self, constructor, box, frame_or_series):
    tz = 'US/Eastern'
    ts = Timestamp('2019', tz=tz)
    if box is None or (frame_or_series is DataFrame and box is dict):
        msg = 'Cannot unbox tzaware Timestamp to tznaive dtype'
        err = TypeError
    else:
        msg = 'Cannot convert timezone-aware data to timezone-naive dtype. Use pd.Series\\(values\\).dt.tz_localize\\(None\\) instead.'
        err = ValueError
    with pytest.raises(err, match=msg):
        constructor(ts, dtype='M8[ns]')