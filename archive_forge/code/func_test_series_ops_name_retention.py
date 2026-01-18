from datetime import (
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import IncompatibleFrequency
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.core.computation.check import NUMEXPR_INSTALLED
@pytest.mark.parametrize('box', [list, tuple, np.array, Index, Series, pd.array])
@pytest.mark.parametrize('flex', [True, False])
def test_series_ops_name_retention(self, flex, box, names, all_binary_operators):
    op = all_binary_operators
    left = Series(range(10), name=names[0])
    right = Series(range(10), name=names[1])
    name = op.__name__.strip('_')
    is_logical = name in ['and', 'rand', 'xor', 'rxor', 'or', 'ror']
    msg = 'Logical ops \\(and, or, xor\\) between Pandas objects and dtype-less sequences'
    warn = None
    if box in [list, tuple] and is_logical:
        warn = FutureWarning
    right = box(right)
    if flex:
        if is_logical:
            return
        result = getattr(left, name)(right)
    else:
        with tm.assert_produces_warning(warn, match=msg):
            result = op(left, right)
    assert isinstance(result, Series)
    if box in [Index, Series]:
        assert result.name is names[2] or result.name == names[2]
    else:
        assert result.name is names[0] or result.name == names[0]