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
@pytest.mark.parametrize('axes', [0, 1, (0, 1), (1, 0), ((1, 0), (2, 1)), ((1, 2), (2, 0)), ((2, 0), (1, 2))])
def test_tensordot_2(axes):
    x = np.arange(4 * 4 * 4).reshape((4, 4, 4))
    y = da.from_array(x, chunks=2)
    assert_eq(da.tensordot(y, y, axes=axes), np.tensordot(x, x, axes=axes))