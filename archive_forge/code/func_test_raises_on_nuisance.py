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
def test_raises_on_nuisance(df):
    grouped = df.groupby('A')
    msg = re.escape('agg function failed [how->mean,dtype->')
    with pytest.raises(TypeError, match=msg):
        grouped.agg('mean')
    with pytest.raises(TypeError, match=msg):
        grouped.mean()
    df = df.loc[:, ['A', 'C', 'D']]
    df['E'] = datetime.now()
    grouped = df.groupby('A')
    msg = 'datetime64 type does not support sum operations'
    with pytest.raises(TypeError, match=msg):
        grouped.agg('sum')
    with pytest.raises(TypeError, match=msg):
        grouped.sum()
    depr_msg = 'DataFrame.groupby with axis=1 is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=depr_msg):
        grouped = df.groupby({'A': 0, 'C': 0, 'D': 1, 'E': 1}, axis=1)
    msg = "does not support reduction 'sum'"
    with pytest.raises(TypeError, match=msg):
        grouped.agg(lambda x: x.sum(0, numeric_only=False))