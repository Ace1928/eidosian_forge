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
def test_order_of_appearance_dt64tz(self, unit):
    dti = DatetimeIndex([Timestamp('20160101', tz='US/Eastern'), Timestamp('20160101', tz='US/Eastern')]).as_unit(unit)
    result = pd.unique(dti)
    expected = DatetimeIndex(['2016-01-01 00:00:00'], dtype=f'datetime64[{unit}, US/Eastern]', freq=None)
    tm.assert_index_equal(result, expected)