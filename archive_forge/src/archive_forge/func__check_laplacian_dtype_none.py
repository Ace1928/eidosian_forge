import pytest
import numpy as np
from numpy.testing import assert_allclose
from pytest import raises as assert_raises
from scipy import sparse
from scipy.sparse import csgraph
from scipy._lib._util import np_long, np_ulong
def _check_laplacian_dtype_none(A, desired_L, desired_d, normed, use_out_degree, copy, dtype, arr_type):
    mat = arr_type(A, dtype=dtype)
    L, d = csgraph.laplacian(mat, normed=normed, return_diag=True, use_out_degree=use_out_degree, copy=copy, dtype=None)
    if normed and check_int_type(mat):
        assert L.dtype == np.float64
        assert d.dtype == np.float64
        _assert_allclose_sparse(L, desired_L, atol=1e-12)
        _assert_allclose_sparse(d, desired_d, atol=1e-12)
    else:
        assert L.dtype == dtype
        assert d.dtype == dtype
        desired_L = np.asarray(desired_L).astype(dtype)
        desired_d = np.asarray(desired_d).astype(dtype)
        _assert_allclose_sparse(L, desired_L, atol=1e-12)
        _assert_allclose_sparse(d, desired_d, atol=1e-12)
    if not copy:
        if not (normed and check_int_type(mat)):
            if type(mat) is np.ndarray:
                assert_allclose(L, mat)
            elif mat.format == 'coo':
                _assert_allclose_sparse(L, mat)