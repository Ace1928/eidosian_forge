import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_slice_locs_na_raises(self):
    index = Index([np.nan, 1, 2])
    with pytest.raises(KeyError, match=''):
        index.slice_locs(start=1.5)
    with pytest.raises(KeyError, match=''):
        index.slice_locs(end=1.5)