import numpy as np
from numpy.testing import (
import numpy.ma.testutils as matest
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def test_triinterpcubic_cg_solver():

    def poisson_sparse_matrix(n, m):
        """
        Return the sparse, (n*m, n*m) matrix in coo format resulting from the
        discretisation of the 2-dimensional Poisson equation according to a
        finite difference numerical scheme on a uniform (n, m) grid.
        """
        l = m * n
        rows = np.concatenate([np.arange(l, dtype=np.int32), np.arange(l - 1, dtype=np.int32), np.arange(1, l, dtype=np.int32), np.arange(l - n, dtype=np.int32), np.arange(n, l, dtype=np.int32)])
        cols = np.concatenate([np.arange(l, dtype=np.int32), np.arange(1, l, dtype=np.int32), np.arange(l - 1, dtype=np.int32), np.arange(n, l, dtype=np.int32), np.arange(l - n, dtype=np.int32)])
        vals = np.concatenate([4 * np.ones(l, dtype=np.float64), -np.ones(l - 1, dtype=np.float64), -np.ones(l - 1, dtype=np.float64), -np.ones(l - n, dtype=np.float64), -np.ones(l - n, dtype=np.float64)])
        vals[l:2 * l - 1][m - 1::m] = 0.0
        vals[2 * l - 1:3 * l - 2][m - 1::m] = 0.0
        return (vals, rows, cols, (n * m, n * m))
    n, m = (12, 4)
    mat = mtri._triinterpolate._Sparse_Matrix_coo(*poisson_sparse_matrix(n, m))
    mat.compress_csc()
    mat_dense = mat.to_dense()
    for itest in range(n * m):
        b = np.zeros(n * m, dtype=np.float64)
        b[itest] = 1.0
        x, _ = mtri._triinterpolate._cg(A=mat, b=b, x0=np.zeros(n * m), tol=1e-10)
        assert_array_almost_equal(np.dot(mat_dense, x), b)
    i_zero, j_zero = (12, 49)
    vals, rows, cols, _ = poisson_sparse_matrix(n, m)
    rows = rows + 1 * (rows >= i_zero) + 1 * (rows >= j_zero)
    cols = cols + 1 * (cols >= i_zero) + 1 * (cols >= j_zero)
    rows = np.concatenate([rows, [i_zero, i_zero - 1, j_zero, j_zero - 1]])
    cols = np.concatenate([cols, [i_zero - 1, i_zero, j_zero - 1, j_zero]])
    vals = np.concatenate([vals, [1.0, 1.0, 1.0, 1.0]])
    mat = mtri._triinterpolate._Sparse_Matrix_coo(vals, rows, cols, (n * m + 2, n * m + 2))
    mat.compress_csc()
    mat_dense = mat.to_dense()
    for itest in range(n * m + 2):
        b = np.zeros(n * m + 2, dtype=np.float64)
        b[itest] = 1.0
        x, _ = mtri._triinterpolate._cg(A=mat, b=b, x0=np.ones(n * m + 2), tol=1e-10)
        assert_array_almost_equal(np.dot(mat_dense, x), b)
    vals = np.ones(17, dtype=np.float64)
    rows = np.array([0, 1, 2, 0, 0, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1], dtype=np.int32)
    cols = np.array([0, 1, 2, 1, 1, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2], dtype=np.int32)
    dim = (3, 3)
    mat = mtri._triinterpolate._Sparse_Matrix_coo(vals, rows, cols, dim)
    mat.compress_csc()
    mat_dense = mat.to_dense()
    assert_array_almost_equal(mat_dense, np.array([[1.0, 2.0, 0.0], [2.0, 1.0, 5.0], [0.0, 5.0, 1.0]], dtype=np.float64))