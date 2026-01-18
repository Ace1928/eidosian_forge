from __future__ import annotations
import itertools
import warnings
import pytest
from tlz import merge
import dask
import dask.array as da
from dask import config
from dask.array.chunk import getitem
from dask.array.slicing import (
from dask.array.utils import assert_eq, same_keys
@pytest.mark.parametrize('shape', [(2,), (2, 3), (2, 3, 5)])
@pytest.mark.parametrize('index', [(Ellipsis,), (None, Ellipsis), (Ellipsis, None), (None, Ellipsis, None)])
def test_slicing_with_Nones(shape, index):
    x = np.random.default_rng().random(shape)
    d = da.from_array(x, chunks=shape)
    assert_eq(x[index], d[index])