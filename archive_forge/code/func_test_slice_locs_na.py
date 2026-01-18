import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_slice_locs_na(self):
    index = Index([np.nan, 1, 2])
    assert index.slice_locs(1) == (1, 3)
    assert index.slice_locs(np.nan) == (0, 3)
    index = Index([0, np.nan, np.nan, 1, 2])
    assert index.slice_locs(np.nan) == (1, 5)