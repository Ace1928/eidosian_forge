import operator
from numpy.core.multiarray import normalize_axis_index
import cupy
from cupyx.scipy import sparse
from cupyx.scipy.sparse.linalg import spsolve
from cupyx.scipy.interpolate._bspline import (
def _make_periodic_spline(x, y, t, k, axis):
    n = x.size
    matr = BSpline.design_matrix(x, t, k)
    temp = cupy.zeros(2 * (2 * k + 1), dtype=float)
    num_c = 1
    dummy_c = cupy.empty((t.size - k - 1, num_c), dtype=float)
    out = cupy.empty((2, 1), dtype=dummy_c.dtype)
    d_boor_kernel = _get_module_func(D_BOOR_MODULE, 'd_boor', dummy_c)
    x0 = cupy.r_[x[0], x[-1]]
    intervals_bc = cupy.array([k, n + k - 1], dtype=cupy.int64)
    rows = cupy.zeros((k - 1, n + k - 1), dtype=float)
    for m in range(k - 1):
        d_boor_kernel((1,), (2,), (t, dummy_c, k, m + 1, x0, intervals_bc, out, temp, num_c, 0, 2))
        rows[m, :k + 1] = temp[:k + 1]
        rows[m, -k:] -= temp[2 * k + 1:2 * k + 1 + k + 1][:-1]
    matr_csr = sparse.vstack([sparse.csr_matrix(rows), matr])
    extradim = prod(y.shape[1:])
    rhs = cupy.empty((n + k - 1, extradim), dtype=float)
    rhs[:k - 1, :] = 0
    rhs[k - 1:, :] = y.reshape(n, 0) if y.size == 0 else y.reshape((-1, extradim))
    coef = spsolve(matr_csr, rhs)
    coef = cupy.ascontiguousarray(coef.reshape((n + k - 1,) + y.shape[1:]))
    return BSpline.construct_fast(t, coef, k, extrapolate='periodic', axis=axis)