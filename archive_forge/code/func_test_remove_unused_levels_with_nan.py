import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.frozen import FrozenList
def test_remove_unused_levels_with_nan():
    idx = Index([(1, np.nan), (3, 4)]).rename(['id1', 'id2'])
    idx = idx.set_levels(['a', np.nan], level='id1')
    idx = idx.remove_unused_levels()
    result = idx.levels
    expected = FrozenList([['a', np.nan], [4]])
    assert str(result) == str(expected)