import numpy as np
import pytest
from pandas.compat import PY311
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_set_levels_pos_args_removal():
    idx = MultiIndex.from_tuples([(1, 'one'), (3, 'one')], names=['foo', 'bar'])
    with pytest.raises(TypeError, match='positional arguments'):
        idx.set_levels(['a', 'b', 'c'], 0)
    with pytest.raises(TypeError, match='positional arguments'):
        idx.set_codes([[0, 1], [1, 0]], 0)