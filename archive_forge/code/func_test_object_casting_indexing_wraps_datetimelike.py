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
def test_object_casting_indexing_wraps_datetimelike(using_array_manager):
    df = DataFrame({'A': [1, 2], 'B': date_range('2000', periods=2), 'C': pd.timedelta_range('1 Day', periods=2)})
    ser = df.loc[0]
    assert isinstance(ser.values[1], Timestamp)
    assert isinstance(ser.values[2], pd.Timedelta)
    ser = df.iloc[0]
    assert isinstance(ser.values[1], Timestamp)
    assert isinstance(ser.values[2], pd.Timedelta)
    ser = df.xs(0, axis=0)
    assert isinstance(ser.values[1], Timestamp)
    assert isinstance(ser.values[2], pd.Timedelta)
    if using_array_manager:
        return
    mgr = df._mgr
    mgr._rebuild_blknos_and_blklocs()
    arr = mgr.fast_xs(0).array
    assert isinstance(arr[1], Timestamp)
    assert isinstance(arr[2], pd.Timedelta)
    blk = mgr.blocks[mgr.blknos[1]]
    assert blk.dtype == 'M8[ns]'
    val = blk.iget((0, 0))
    assert isinstance(val, Timestamp)
    blk = mgr.blocks[mgr.blknos[2]]
    assert blk.dtype == 'm8[ns]'
    val = blk.iget((0, 0))
    assert isinstance(val, pd.Timedelta)