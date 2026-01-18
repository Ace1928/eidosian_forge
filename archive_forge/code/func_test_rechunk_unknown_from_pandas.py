from __future__ import annotations
import warnings
from itertools import product
import pytest
import math
import dask
import dask.array as da
from dask.array.rechunk import (
from dask.array.utils import assert_eq
from dask.utils import funcname
def test_rechunk_unknown_from_pandas():
    dd = pytest.importorskip('dask.dataframe')
    pd = pytest.importorskip('pandas')
    arr = np.random.default_rng().standard_normal((50, 10))
    x = dd.from_pandas(pd.DataFrame(arr), 2).values
    result = x.rechunk((None, (5, 5)))
    assert np.isnan(x.chunks[0]).all()
    assert np.isnan(result.chunks[0]).all()
    assert result.chunks[1] == (5, 5)
    expected = da.from_array(arr, chunks=((25, 25), (10,))).rechunk((None, (5, 5)))
    assert_eq(result, expected)