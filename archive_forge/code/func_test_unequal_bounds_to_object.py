import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import (
from pandas.core.indexers.objects import (
from pandas.tseries.offsets import BusinessDay
def test_unequal_bounds_to_object():

    class CustomIndexer(BaseIndexer):

        def get_window_bounds(self, num_values, min_periods, center, closed, step):
            return (np.array([1]), np.array([2]))
    indexer = CustomIndexer()
    roll = Series([1, 1]).rolling(indexer)
    match = 'start and end'
    with pytest.raises(ValueError, match=match):
        roll.mean()
    with pytest.raises(ValueError, match=match):
        next(iter(roll))
    with pytest.raises(ValueError, match=match):
        roll.corr(pairwise=True)
    with pytest.raises(ValueError, match=match):
        roll.cov(pairwise=True)