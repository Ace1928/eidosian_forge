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
@pytest.mark.parametrize('dtype', ['f2', 'f4', 'f8', 'f16', 'c8', 'c16', 'c32'])
@pytest.mark.parametrize('u_based', [True, False])
def test_svd_flip_sign(dtype, u_based):
    try:
        x = np.array([[1, -1, 1, -1], [1, -1, 1, -1], [-1, 1, 1, -1], [-1, 1, 1, -1]], dtype=dtype)
    except TypeError:
        pytest.skip('128-bit floats not supported by NumPy')
    u, v = svd_flip(x, x.T, u_based_decision=u_based)
    assert u.dtype == x.dtype
    assert v.dtype == x.dtype
    y = x.copy()
    y[:, -1] *= y.dtype.type(-1)
    assert_eq(u, y)
    assert_eq(v, y.T)