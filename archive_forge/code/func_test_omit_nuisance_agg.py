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
@pytest.mark.parametrize('agg_function', ['sum', 'mean', 'prod', 'std', 'var', 'sem', 'median'])
@pytest.mark.parametrize('numeric_only', [True, False])
def test_omit_nuisance_agg(df, agg_function, numeric_only):
    grouped = df.groupby('A')
    no_drop_nuisance = ('var', 'std', 'sem', 'mean', 'prod', 'median')
    if agg_function in no_drop_nuisance and (not numeric_only):
        if agg_function in ('std', 'sem'):
            klass = ValueError
            msg = "could not convert string to float: 'one'"
        else:
            klass = TypeError
            msg = re.escape(f'agg function failed [how->{agg_function},dtype->')
        with pytest.raises(klass, match=msg):
            getattr(grouped, agg_function)(numeric_only=numeric_only)
    else:
        result = getattr(grouped, agg_function)(numeric_only=numeric_only)
        if not numeric_only and agg_function == 'sum':
            columns = ['A', 'B', 'C', 'D']
        else:
            columns = ['A', 'C', 'D']
        expected = getattr(df.loc[:, columns].groupby('A'), agg_function)(numeric_only=numeric_only)
        tm.assert_frame_equal(result, expected)