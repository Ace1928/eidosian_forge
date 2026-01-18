from __future__ import annotations
import sys
import pytest
import numpy as np
import scipy.linalg
from packaging.version import parse as parse_version
import dask.array as da
from dask.array.linalg import qr, sfqr, svd, svd_compressed, tsqr
from dask.array.numpy_compat import _np_version
from dask.array.utils import assert_eq, same_keys, svd_flip
@pytest.mark.parametrize('m', [5, 10, 15, 20])
@pytest.mark.parametrize('n', [5, 10, 15, 20])
@pytest.mark.parametrize('k', [5])
@pytest.mark.parametrize('chunks', [(5, 10), (10, 5)])
def test_svd_compressed_shapes(m, n, k, chunks):
    x = da.random.default_rng().random(size=(m, n), chunks=chunks)
    u, s, v = svd_compressed(x, k, n_power_iter=1, compute=True, seed=1)
    u, s, v = da.compute(u, s, v)
    r = min(m, n, k)
    assert u.shape == (m, r)
    assert s.shape == (r,)
    assert v.shape == (r, n)