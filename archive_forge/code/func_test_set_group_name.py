from datetime import datetime
import decimal
from decimal import Decimal
import re
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BooleanArray
import pandas.core.common as com
@pytest.mark.parametrize('grouper', ['A', ['A', 'B']])
def test_set_group_name(df, grouper, using_infer_string):

    def f(group):
        assert group.name is not None
        return group

    def freduce(group):
        assert group.name is not None
        if using_infer_string and grouper == 'A' and is_string_dtype(group.dtype):
            with pytest.raises(TypeError, match='does not support'):
                group.sum()
        else:
            return group.sum()

    def freducex(x):
        return freduce(x)
    grouped = df.groupby(grouper, group_keys=False)
    msg = 'DataFrameGroupBy.apply operated on the grouping columns'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        grouped.apply(f)
    grouped.aggregate(freduce)
    grouped.aggregate({'C': freduce, 'D': freduce})
    grouped.transform(f)
    grouped['C'].apply(f)
    grouped['C'].aggregate(freduce)
    grouped['C'].aggregate([freduce, freducex])
    grouped['C'].transform(f)