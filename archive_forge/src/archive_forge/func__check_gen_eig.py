import itertools
import platform
import sys
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.linalg import (eig, eigvals, lu, svd, svdvals, cholesky, qr,
from scipy.linalg.lapack import (dgbtrf, dgbtrs, zgbtrf, zgbtrs, dsbev,
from scipy.linalg._misc import norm
from scipy.linalg._decomp_qz import _select_function
from scipy.stats import ortho_group
from numpy import (array, diag, full, linalg, argsort, zeros, arange,
from scipy.linalg._testutils import assert_no_overwrite
from scipy.sparse._sputils import matrix
from scipy._lib._testutils import check_free_memory
from scipy.linalg.blas import HAS_ILP64
def _check_gen_eig(self, A, B):
    if B is not None:
        A, B = (asarray(A), asarray(B))
        B0 = B
    else:
        A = asarray(A)
        B0 = B
        B = np.eye(*A.shape)
    msg = f'\n{A!r}\n{B!r}'
    w, vr = eig(A, B0, homogeneous_eigvals=True)
    wt = eigvals(A, B0, homogeneous_eigvals=True)
    val1 = A @ vr * w[1, :]
    val2 = B @ vr * w[0, :]
    for i in range(val1.shape[1]):
        assert_allclose(val1[:, i], val2[:, i], rtol=1e-13, atol=1e-13, err_msg=msg)
    if B0 is None:
        assert_allclose(w[1, :], 1)
        assert_allclose(wt[1, :], 1)
    perm = np.lexsort(w)
    permt = np.lexsort(wt)
    assert_allclose(w[:, perm], wt[:, permt], atol=1e-07, rtol=1e-07, err_msg=msg)
    length = np.empty(len(vr))
    for i in range(len(vr)):
        length[i] = norm(vr[:, i])
    assert_allclose(length, np.ones(length.size), err_msg=msg, atol=1e-07, rtol=1e-07)
    beta_nonzero = w[1, :] != 0
    wh = w[0, beta_nonzero] / w[1, beta_nonzero]
    w, vr = eig(A, B0)
    wt = eigvals(A, B0)
    val1 = A @ vr
    val2 = B @ vr * w
    res = val1 - val2
    for i in range(res.shape[1]):
        if np.all(isfinite(res[:, i])):
            assert_allclose(res[:, i], 0, rtol=1e-13, atol=1e-13, err_msg=msg)
    w_fin = w[isfinite(w)]
    wt_fin = wt[isfinite(wt)]
    perm = argsort(clear_fuss(w_fin))
    permt = argsort(clear_fuss(wt_fin))
    assert_allclose(w[perm], wt[permt], atol=1e-07, rtol=1e-07, err_msg=msg)
    length = np.empty(len(vr))
    for i in range(len(vr)):
        length[i] = norm(vr[:, i])
    assert_allclose(length, np.ones(length.size), err_msg=msg)
    assert_allclose(sort(wh), sort(w[np.isfinite(w)]))