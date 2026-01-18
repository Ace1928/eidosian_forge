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
@pytest.mark.parametrize('iterator', [('power', 2), ('QR', 2)])
def test_svd_compressed(iterator):
    m, n = (100, 50)
    r = 5
    a = da.random.default_rng().random((m, n), chunks=(m, n))
    u, s, vt = svd_compressed(a, 2 * r, iterator=iterator[0], n_power_iter=iterator[1], seed=4321)
    s_true = scipy.linalg.svd(a.compute(), compute_uv=False)
    norm = scipy.linalg.norm((a - u[:, :r] * s[:r] @ vt[:r, :]).compute(), 2)
    frac = norm / s_true[r + 1] - 1
    tol = 0.4
    assert frac < tol
    assert_eq(np.eye(r, r), da.dot(u[:, :r].T, u[:, :r]))
    assert_eq(np.eye(r, r), da.dot(vt[:r, :], vt[:r, :].T))