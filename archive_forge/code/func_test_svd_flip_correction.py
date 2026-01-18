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
@pytest.mark.parametrize('shape', [(10, 20), (10, 10), (20, 10)])
@pytest.mark.parametrize('chunks', [(-1, -1), (10, -1), (-1, 10)])
@pytest.mark.parametrize('dtype', ['f4', 'f8'])
def test_svd_flip_correction(shape, chunks, dtype):
    x = da.random.default_rng().random(size=shape, chunks=chunks).astype(dtype)
    u, s, v = da.linalg.svd(x)
    decimal = 9 if np.dtype(dtype).itemsize > 4 else 6
    uf, vf = svd_flip(u, v)
    assert uf.dtype == u.dtype
    assert vf.dtype == v.dtype
    np.testing.assert_almost_equal(np.asarray(np.dot(uf * s, vf)), x, decimal=decimal)
    uc, vc = svd_flip(*da.compute(u, v))
    assert uc.dtype == u.dtype
    assert vc.dtype == v.dtype
    np.testing.assert_almost_equal(np.asarray(np.dot(uc * s, vc)), x, decimal=decimal)