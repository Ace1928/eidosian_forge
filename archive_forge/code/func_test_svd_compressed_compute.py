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
@pytest.mark.parametrize('iterator', ['power', 'QR'])
def test_svd_compressed_compute(iterator):
    x = da.ones((100, 100), chunks=(10, 10))
    u, s, v = da.linalg.svd_compressed(x, k=2, iterator=iterator, n_power_iter=1, compute=True, seed=123)
    uu, ss, vv = da.linalg.svd_compressed(x, k=2, iterator=iterator, n_power_iter=1, seed=123)
    assert len(v.dask) < len(vv.dask)
    assert_eq(v, vv)