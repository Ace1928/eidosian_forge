from copy import (
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
from pandas import (
import pandas._testing as tm
def test_split_compat(self, frame_or_series):
    o = construct(frame_or_series, shape=10)
    with tm.assert_produces_warning(FutureWarning, match=".swapaxes' is deprecated", check_stacklevel=False):
        assert len(np.array_split(o, 5)) == 5
        assert len(np.array_split(o, 2)) == 2