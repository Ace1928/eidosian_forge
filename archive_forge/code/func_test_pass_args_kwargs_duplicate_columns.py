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
@pytest.mark.parametrize('as_index', [True, False])
def test_pass_args_kwargs_duplicate_columns(tsframe, as_index):
    tsframe.columns = ['A', 'B', 'A', 'C']
    gb = tsframe.groupby(lambda x: x.month, as_index=as_index)
    warn = None if as_index else FutureWarning
    msg = 'A grouping .* was excluded from the result'
    with tm.assert_produces_warning(warn, match=msg):
        res = gb.agg(np.percentile, 80, axis=0)
    ex_data = {1: tsframe[tsframe.index.month == 1].quantile(0.8), 2: tsframe[tsframe.index.month == 2].quantile(0.8)}
    expected = DataFrame(ex_data).T
    if not as_index:
        expected.index = Index(range(2))
    tm.assert_frame_equal(res, expected)