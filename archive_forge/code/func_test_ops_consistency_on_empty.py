from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
@pytest.mark.parametrize('method', ['mean', 'median', 'std', 'var'])
def test_ops_consistency_on_empty(self, method):
    result = getattr(Series(dtype=float), method)()
    assert isna(result)
    tdser = Series([], dtype='m8[ns]')
    if method == 'var':
        msg = '|'.join(["operation 'var' not allowed", 'cannot perform var with type timedelta64\\[ns\\]', "does not support reduction 'var'"])
        with pytest.raises(TypeError, match=msg):
            getattr(tdser, method)()
    else:
        result = getattr(tdser, method)()
        assert result is NaT