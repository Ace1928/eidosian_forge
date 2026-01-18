from __future__ import annotations
import numpy as np
import pytest
from numpy.testing import assert_equal
import dask.array as da
from dask.array.core import Array
from dask.array.gufunc import (
from dask.array.utils import assert_eq
def test_preserve_meta_type():
    sparse = pytest.importorskip('sparse')

    def stats(x):
        return (np.sum(x, axis=-1), np.mean(x, axis=-1))
    a = da.random.default_rng().normal(size=(10, 20, 30), chunks=(5, 5, 30))
    a = a.map_blocks(sparse.COO.from_numpy)
    sum, mean = apply_gufunc(stats, '(i)->(),()', a, output_dtypes=2 * (a.dtype,))
    assert isinstance(a._meta, sparse.COO)
    assert isinstance(sum._meta, sparse.COO)
    assert isinstance(mean._meta, sparse.COO)
    assert_eq(sum, sum)
    assert_eq(mean, mean)