from __future__ import annotations
import contextlib
import itertools
import pickle
import sys
import warnings
from numbers import Number
import pytest
import dask
from dask.delayed import delayed
import dask.array as da
from dask.array.numpy_compat import NUMPY_GE_123, NUMPY_GE_200, AxisError
from dask.array.utils import assert_eq, same_keys
@pytest.mark.parametrize('dims', [da.from_array([5, 10]), delayed([5, 10], nout=2)])
@pytest.mark.parametrize('wrap_in_list', [False, True])
def test_ravel_multi_index_delayed_dims(dims, wrap_in_list):
    with pytest.raises(NotImplementedError, match='Dask types are not supported'):
        da.ravel_multi_index((2, 1), [dims[0], dims[1]] if wrap_in_list else dims)