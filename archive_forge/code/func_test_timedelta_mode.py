from datetime import datetime
import struct
import numpy as np
import pytest
from pandas._libs import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
import pandas.core.common as com
def test_timedelta_mode(self):
    exp = Series(['-1 days', '0 days', '1 days'], dtype='timedelta64[ns]')
    ser = Series(['1 days', '-1 days', '0 days'], dtype='timedelta64[ns]')
    tm.assert_extension_array_equal(algos.mode(ser.values), exp._values)
    tm.assert_series_equal(ser.mode(), exp)
    exp = Series(['2 min', '1 day'], dtype='timedelta64[ns]')
    ser = Series(['1 day', '1 day', '-1 day', '-1 day 2 min', '2 min', '2 min'], dtype='timedelta64[ns]')
    tm.assert_extension_array_equal(algos.mode(ser.values), exp._values)
    tm.assert_series_equal(ser.mode(), exp)