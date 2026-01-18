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
def test_rechunk_unknown_raises():
    dd = pytest.importorskip('dask.dataframe')
    x = da.ones(shape=(10, 10), chunks=(5, 5))
    y = dd.from_array(x).values
    with pytest.raises(ValueError, match='Chunks do not add'):
        y.rechunk((None, (5, 5, 5)))
    with pytest.raises(ValueError, match='Chunks must be unchanging'):
        y.rechunk(((np.nan, np.nan, np.nan), (5, 5)))
    with pytest.raises(ValueError, match='Chunks must be unchanging'):
        y.rechunk(((5, 5), (5, 5)))
    with pytest.raises(ValueError, match='Chunks must be unchanging'):
        z = da.concatenate([x, y])
        z.rechunk(((5, 3, 2, np.nan, np.nan), (5, 5)))