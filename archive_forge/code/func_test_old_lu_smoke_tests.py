import pytest
from pytest import raises as assert_raises
import numpy as np
from scipy.linalg import lu, lu_factor, lu_solve, get_lapack_funcs, solve
from numpy.testing import assert_allclose, assert_array_equal
def test_old_lu_smoke_tests(self):
    """Tests from old fortran based lu test suite"""
    a = np.array([[1, 2, 3], [1, 2, 3], [2, 5, 6]])
    p, l, u = lu(a)
    result_lu = np.array([[2.0, 5.0, 6.0], [0.5, -0.5, 0.0], [0.5, 1.0, 0.0]])
    assert_allclose(p, np.rot90(np.eye(3)))
    assert_allclose(l, np.tril(result_lu, k=-1) + np.eye(3))
    assert_allclose(u, np.triu(result_lu))
    a = np.array([[1, 2, 3], [1, 2, 3], [2, 5j, 6]])
    p, l, u = lu(a)
    result_lu = np.array([[2.0, 5j, 6.0], [0.5, 2 - 2.5j, 0.0], [0.5, 1.0, 0.0]])
    assert_allclose(p, np.rot90(np.eye(3)))
    assert_allclose(l, np.tril(result_lu, k=-1) + np.eye(3))
    assert_allclose(u, np.triu(result_lu))
    b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    p, l, u = lu(b)
    assert_allclose(p, np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]))
    assert_allclose(l, np.array([[1, 0, 0], [1 / 7, 1, 0], [4 / 7, 0.5, 1]]))
    assert_allclose(u, np.array([[7, 8, 9], [0, 6 / 7, 12 / 7], [0, 0, 0]]), rtol=0.0, atol=1e-14)
    cb = np.array([[1j, 2j, 3j], [4j, 5j, 6j], [7j, 8j, 9j]])
    p, l, u = lu(cb)
    assert_allclose(p, np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]))
    assert_allclose(l, np.array([[1, 0, 0], [1 / 7, 1, 0], [4 / 7, 0.5, 1]]))
    assert_allclose(u, np.array([[7, 8, 9], [0, 6 / 7, 12 / 7], [0, 0, 0]]) * 1j, rtol=0.0, atol=1e-14)
    hrect = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 12, 12]])
    p, l, u = lu(hrect)
    assert_allclose(p, np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]))
    assert_allclose(l, np.array([[1, 0, 0], [1 / 9, 1, 0], [5 / 9, 0.5, 1]]))
    assert_allclose(u, np.array([[9, 10, 12, 12], [0, 8 / 9, 15 / 9, 24 / 9], [0, 0, -0.5, 0]]), rtol=0.0, atol=1e-14)
    chrect = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 12, 12]]) * 1j
    p, l, u = lu(chrect)
    assert_allclose(p, np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]))
    assert_allclose(l, np.array([[1, 0, 0], [1 / 9, 1, 0], [5 / 9, 0.5, 1]]))
    assert_allclose(u, np.array([[9, 10, 12, 12], [0, 8 / 9, 15 / 9, 24 / 9], [0, 0, -0.5, 0]]) * 1j, rtol=0.0, atol=1e-14)
    vrect = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 12, 12]])
    p, l, u = lu(vrect)
    assert_allclose(p, np.eye(4)[[1, 3, 2, 0], :])
    assert_allclose(l, np.array([[1.0, 0, 0], [0.1, 1, 0], [0.7, -0.5, 1], [0.4, 0.25, 0.5]]))
    assert_allclose(u, np.array([[10, 12, 12], [0, 0.8, 1.8], [0, 0, 1.5]]))
    cvrect = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 12, 12]]) * 1j
    p, l, u = lu(cvrect)
    assert_allclose(p, np.eye(4)[[1, 3, 2, 0], :])
    assert_allclose(l, np.array([[1.0, 0, 0], [0.1, 1, 0], [0.7, -0.5, 1], [0.4, 0.25, 0.5]]))
    assert_allclose(u, np.array([[10, 12, 12], [0, 0.8, 1.8], [0, 0, 1.5]]) * 1j)