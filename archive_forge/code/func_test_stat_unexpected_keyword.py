from copy import (
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
from pandas import (
import pandas._testing as tm
def test_stat_unexpected_keyword(self, frame_or_series):
    obj = construct(frame_or_series, 5)
    starwars = 'Star Wars'
    errmsg = 'unexpected keyword'
    with pytest.raises(TypeError, match=errmsg):
        obj.max(epic=starwars)
    with pytest.raises(TypeError, match=errmsg):
        obj.var(epic=starwars)
    with pytest.raises(TypeError, match=errmsg):
        obj.sum(epic=starwars)
    with pytest.raises(TypeError, match=errmsg):
        obj.any(epic=starwars)