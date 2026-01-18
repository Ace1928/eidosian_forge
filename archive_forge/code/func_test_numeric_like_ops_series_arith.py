import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('op, str_rep', [('__add__', '\\+'), ('__sub__', '-'), ('__mul__', '\\*'), ('__truediv__', '/')])
def test_numeric_like_ops_series_arith(self, op, str_rep):
    s = Series(Categorical([1, 2, 3, 4]))
    msg = f'Series cannot perform the operation {str_rep}|unsupported operand'
    with pytest.raises(TypeError, match=msg):
        getattr(s, op)(2)