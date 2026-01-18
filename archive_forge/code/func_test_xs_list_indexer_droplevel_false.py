import re
import numpy as np
import pytest
from pandas.errors import SettingWithCopyError
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
def test_xs_list_indexer_droplevel_false(self):
    mi = MultiIndex.from_tuples([('x', 'm', 'a'), ('x', 'n', 'b'), ('y', 'o', 'c')])
    df = DataFrame([[1, 2, 3], [4, 5, 6]], columns=mi)
    with pytest.raises(KeyError, match='y'):
        df.xs(('x', 'y'), drop_level=False, axis=1)