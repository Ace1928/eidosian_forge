from datetime import (
import itertools
import re
import numpy as np
import pytest
from pandas._libs.internals import BlockPlacement
from pandas.compat import IS64
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
from pandas.core.internals import (
from pandas.core.internals.blocks import (
@pytest.mark.parametrize('mgr', MANAGERS)
@pytest.mark.parametrize('fill_value', [None, np.nan, 100.0])
def test_reindex_axis(self, fill_value, mgr):

    def assert_reindex_axis_is_ok(mgr, axis, new_labels, fill_value):
        mat = _as_array(mgr)
        indexer = mgr.axes[axis].get_indexer_for(new_labels)
        reindexed = mgr.reindex_axis(new_labels, axis, fill_value=fill_value)
        tm.assert_numpy_array_equal(algos.take_nd(mat, indexer, axis, fill_value=fill_value), _as_array(reindexed), check_dtype=False)
        tm.assert_index_equal(reindexed.axes[axis], new_labels)
    for ax in range(mgr.ndim):
        assert_reindex_axis_is_ok(mgr, ax, Index([]), fill_value)
        assert_reindex_axis_is_ok(mgr, ax, mgr.axes[ax], fill_value)
        assert_reindex_axis_is_ok(mgr, ax, mgr.axes[ax][[0, 0, 0]], fill_value)
        assert_reindex_axis_is_ok(mgr, ax, Index(['foo', 'bar', 'baz']), fill_value)
        assert_reindex_axis_is_ok(mgr, ax, Index(['foo', mgr.axes[ax][0], 'baz']), fill_value)
        if mgr.shape[ax] >= 3:
            assert_reindex_axis_is_ok(mgr, ax, mgr.axes[ax][:-3], fill_value)
            assert_reindex_axis_is_ok(mgr, ax, mgr.axes[ax][-3::-1], fill_value)
            assert_reindex_axis_is_ok(mgr, ax, mgr.axes[ax][[0, 1, 2, 0, 1, 2]], fill_value)