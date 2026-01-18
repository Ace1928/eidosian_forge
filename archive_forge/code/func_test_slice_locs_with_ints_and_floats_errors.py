import re
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('query', [[0, 1], [0, 2], [0, 3], [0, 4]])
@pytest.mark.parametrize('tuples', [[(0, 2), (1, 3), (2, 4)], [(2, 4), (1, 3), (0, 2)], [(0, 2), (0, 2), (2, 4)], [(0, 2), (2, 4), (0, 2)], [(0, 2), (0, 2), (2, 4), (1, 3)]])
def test_slice_locs_with_ints_and_floats_errors(self, tuples, query):
    start, stop = query
    index = IntervalIndex.from_tuples(tuples)
    with pytest.raises(KeyError, match="'can only get slices from an IntervalIndex if bounds are non-overlapping and all monotonic increasing or decreasing'"):
        index.slice_locs(start, stop)