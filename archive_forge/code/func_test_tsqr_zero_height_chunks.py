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
def test_tsqr_zero_height_chunks():
    m_q = 10
    n_q = 5
    m_r = 5
    n_r = 5
    mat = np.random.default_rng().random((10, 5))
    x = da.from_array(mat, chunks=((4, 0, 1, 0, 5), (5,)))
    q, r = da.linalg.qr(x)
    assert_eq((m_q, n_q), q.shape)
    assert_eq((m_r, n_r), r.shape)
    assert_eq(mat, da.dot(q, r))
    assert_eq(np.eye(n_q, n_q), da.dot(q.T, q))
    assert_eq(r, da.triu(r.rechunk(r.shape[0])))
    mat2 = np.vstack([mat, -np.ones((10, 5))])
    v2 = mat2[:, 0]
    x2 = da.from_array(mat2, chunks=5)
    c = da.from_array(v2, chunks=5)
    x = x2[c >= 0, :]
    q, r = da.linalg.qr(x)
    q = q.compute()
    r = r.compute()
    assert_eq((m_q, n_q), q.shape)
    assert_eq((m_r, n_r), r.shape)
    assert_eq(mat, np.dot(q, r))
    assert_eq(np.eye(n_q, n_q), np.dot(q.T, q))
    assert_eq(r, np.triu(r))