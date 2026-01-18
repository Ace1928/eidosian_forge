from copy import (
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
from pandas import (
import pandas._testing as tm
def test_stat_non_defaults_args(self, frame_or_series):
    obj = construct(frame_or_series, 5)
    out = np.array([0])
    errmsg = "the 'out' parameter is not supported"
    with pytest.raises(ValueError, match=errmsg):
        obj.max(out=out)
    with pytest.raises(ValueError, match=errmsg):
        obj.var(out=out)
    with pytest.raises(ValueError, match=errmsg):
        obj.sum(out=out)
    with pytest.raises(ValueError, match=errmsg):
        obj.any(out=out)