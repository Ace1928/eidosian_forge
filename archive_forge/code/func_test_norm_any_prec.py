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
@pytest.mark.xfail(_np_version < parse_version('1.23'), reason='https://github.com/numpy/numpy/pull/17709', strict=False)
@pytest.mark.parametrize('precision', ['single', 'double'])
@pytest.mark.parametrize('isreal', [True, False])
@pytest.mark.parametrize('keepdims', [False, True])
@pytest.mark.parametrize('norm', [None, 1, -1, np.inf, -np.inf])
def test_norm_any_prec(norm, keepdims, precision, isreal):
    shape, chunks, axis = ((5,), (2,), None)
    precs_r = {'single': 'float32', 'double': 'float64'}
    precs_c = {'single': 'complex64', 'double': 'complex128'}
    dtype = precs_r[precision] if isreal else precs_c[precision]
    a = np.random.default_rng().random(shape).astype(dtype)
    d = da.from_array(a, chunks=chunks)
    d_a = np.linalg.norm(a, ord=norm, axis=axis, keepdims=keepdims)
    d_r = da.linalg.norm(d, ord=norm, axis=axis, keepdims=keepdims)
    assert d_r.dtype == precs_r[precision]
    assert d_r.dtype == d_a.dtype