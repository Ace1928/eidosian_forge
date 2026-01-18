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
@pytest.mark.slow
@pytest.mark.xfail(sys.platform == 'darwin' and _np_version < parse_version('1.22'), reason='https://github.com/dask/dask/issues/7189', strict=False)
@pytest.mark.parametrize('shape, chunks', [[(5,), (2,)], [(5, 3), (2, 2)], [(4, 5, 3), (2, 2, 2)], [(4, 5, 2, 3), (2, 2, 2, 2)], [(2, 5, 2, 4, 3), (2, 2, 2, 2, 2)]])
@pytest.mark.parametrize('norm', [None, 1, -1, np.inf, -np.inf])
@pytest.mark.parametrize('keepdims', [False, True])
def test_norm_any_slice(shape, chunks, norm, keepdims):
    a = np.random.default_rng().random(shape)
    d = da.from_array(a, chunks=chunks)
    for firstaxis in range(len(shape)):
        for secondaxis in range(len(shape)):
            if firstaxis != secondaxis:
                axis = (firstaxis, secondaxis)
            else:
                axis = firstaxis
            a_r = np.linalg.norm(a, ord=norm, axis=axis, keepdims=keepdims)
            d_r = da.linalg.norm(d, ord=norm, axis=axis, keepdims=keepdims)
            assert_eq(a_r, d_r)