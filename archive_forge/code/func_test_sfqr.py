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
@pytest.mark.parametrize('m,n,chunks,error_type', [(20, 10, 10, ValueError), (20, 10, (3, 10), ValueError), (20, 10, ((8, 4, 8), 10), ValueError), (40, 10, ((15, 5, 5, 8, 7), 10), ValueError), (128, 2, (16, 2), ValueError), (129, 2, (16, 2), ValueError), (130, 2, (16, 2), ValueError), (131, 2, (16, 2), ValueError), (300, 10, (40, 10), ValueError), (300, 10, (30, 10), ValueError), (300, 10, (20, 10), ValueError), (10, 5, 10, None), (5, 10, 10, None), (10, 10, 10, None), (10, 40, (10, 10), None), (10, 40, (10, 15), None), (10, 40, (10, (15, 5, 5, 8, 7)), None), (20, 20, 10, ValueError)])
def test_sfqr(m, n, chunks, error_type):
    mat = np.random.default_rng().random((m, n))
    data = da.from_array(mat, chunks=chunks, name='A')
    m_q = m
    n_q = min(m, n)
    m_r = n_q
    n_r = n
    m_qtq = n_q
    if error_type is None:
        q, r = sfqr(data)
        assert_eq((m_q, n_q), q.shape)
        assert_eq((m_r, n_r), r.shape)
        assert_eq(mat, da.dot(q, r))
        assert_eq(np.eye(m_qtq, m_qtq), da.dot(q.T, q))
        assert_eq(r, da.triu(r.rechunk(r.shape[0])))
    else:
        with pytest.raises(error_type):
            q, r = sfqr(data)