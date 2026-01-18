from __future__ import annotations
import numpy as np
import pytest
import dask.array as da
from dask.array.tests.test_dispatch import EncapsulateNDArray, WrappedArray
from dask.array.utils import assert_eq
@pytest.mark.parametrize('func', [lambda x: np.concatenate([x, x, x]), lambda x: np.cov(x, x), lambda x: np.dot(x, x), lambda x: np.dstack((x, x)), lambda x: np.flip(x, axis=0), lambda x: np.hstack((x, x)), lambda x: np.matmul(x, x), lambda x: np.mean(x), lambda x: np.stack([x, x]), lambda x: np.sum(x), lambda x: np.var(x), lambda x: np.vstack((x, x)), lambda x: np.linalg.norm(x)])
def test_unregistered_func(func):
    x = EncapsulateNDArray(np.random.default_rng().random((100, 100)))
    y = da.from_array(x, chunks=(50, 50))
    assert_eq(x, y, check_meta=False, check_type=False)
    xx = func(x)
    yy = func(y)
    assert_eq(xx, yy, check_meta=False, check_type=False)