from __future__ import annotations
import pytest
import numpy as np
import dask
import dask.array as da
from dask.array.core import Array
from dask.array.utils import assert_eq
from dask.multiprocessing import _dumps, _loads
from dask.utils import key_split
def test_randomstate_kwargs():
    cupy = pytest.importorskip('cupy')
    rs = da.random.RandomState(RandomState=cupy.random.RandomState)
    x = rs.standard_normal((10, 5), dtype=np.float32)
    assert x.dtype == np.float32
    rs = da.random.default_rng(cupy.random.default_rng())
    x = rs.standard_normal((10, 5), dtype=np.float32)
    assert x.dtype == np.float32