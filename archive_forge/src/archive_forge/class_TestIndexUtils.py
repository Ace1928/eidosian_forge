from collections import defaultdict
from datetime import datetime
from functools import partial
import math
import operator
import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.api import (
class TestIndexUtils:

    @pytest.mark.parametrize('data, names, expected', [([[1, 2, 3]], None, Index([1, 2, 3])), ([[1, 2, 3]], ['name'], Index([1, 2, 3], name='name')), ([['a', 'a'], ['c', 'd']], None, MultiIndex([['a'], ['c', 'd']], [[0, 0], [0, 1]])), ([['a', 'a'], ['c', 'd']], ['L1', 'L2'], MultiIndex([['a'], ['c', 'd']], [[0, 0], [0, 1]], names=['L1', 'L2']))])
    def test_ensure_index_from_sequences(self, data, names, expected):
        result = ensure_index_from_sequences(data, names)
        tm.assert_index_equal(result, expected)

    def test_ensure_index_mixed_closed_intervals(self):
        intervals = [pd.Interval(0, 1, closed='left'), pd.Interval(1, 2, closed='right'), pd.Interval(2, 3, closed='neither'), pd.Interval(3, 4, closed='both')]
        result = ensure_index(intervals)
        expected = Index(intervals, dtype=object)
        tm.assert_index_equal(result, expected)

    def test_ensure_index_uint64(self):
        values = [0, np.iinfo(np.uint64).max]
        result = ensure_index(values)
        assert list(result) == values
        expected = Index(values, dtype='uint64')
        tm.assert_index_equal(result, expected)

    def test_get_combined_index(self):
        result = _get_combined_index([])
        expected = Index([])
        tm.assert_index_equal(result, expected)