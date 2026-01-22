from decimal import Decimal
import numpy as np
import pytest
from pandas._libs.missing import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import Index
import pandas._testing as tm
class TestGetIndexer:

    @pytest.mark.parametrize('method,expected', [('pad', np.array([-1, 0, 1, 1], dtype=np.intp)), ('backfill', np.array([0, 0, 1, -1], dtype=np.intp))])
    def test_get_indexer_strings(self, method, expected):
        index = Index(['b', 'c'])
        actual = index.get_indexer(['a', 'b', 'c', 'd'], method=method)
        tm.assert_numpy_array_equal(actual, expected)

    def test_get_indexer_strings_raises(self, using_infer_string):
        index = Index(['b', 'c'])
        if using_infer_string:
            import pyarrow as pa
            msg = 'has no kernel'
            with pytest.raises(pa.lib.ArrowNotImplementedError, match=msg):
                index.get_indexer(['a', 'b', 'c', 'd'], method='nearest')
            with pytest.raises(pa.lib.ArrowNotImplementedError, match=msg):
                index.get_indexer(['a', 'b', 'c', 'd'], method='pad', tolerance=2)
            with pytest.raises(pa.lib.ArrowNotImplementedError, match=msg):
                index.get_indexer(['a', 'b', 'c', 'd'], method='pad', tolerance=[2, 2, 2, 2])
        else:
            msg = "unsupported operand type\\(s\\) for -: 'str' and 'str'"
            with pytest.raises(TypeError, match=msg):
                index.get_indexer(['a', 'b', 'c', 'd'], method='nearest')
            with pytest.raises(TypeError, match=msg):
                index.get_indexer(['a', 'b', 'c', 'd'], method='pad', tolerance=2)
            with pytest.raises(TypeError, match=msg):
                index.get_indexer(['a', 'b', 'c', 'd'], method='pad', tolerance=[2, 2, 2, 2])

    def test_get_indexer_with_NA_values(self, unique_nulls_fixture, unique_nulls_fixture2):
        if unique_nulls_fixture is unique_nulls_fixture2:
            return
        arr = np.array([unique_nulls_fixture, unique_nulls_fixture2], dtype=object)
        index = Index(arr, dtype=object)
        result = index.get_indexer(Index([unique_nulls_fixture, unique_nulls_fixture2, 'Unknown'], dtype=object))
        expected = np.array([0, 1, -1], dtype=np.intp)
        tm.assert_numpy_array_equal(result, expected)