import pytest
import numpy as np
from numpy.linalg import norm as npnorm
from numpy.testing import assert_allclose, assert_equal
from pytest import raises as assert_raises
import scipy.sparse
from scipy.sparse.linalg import norm as spnorm
def test_sparray_norm():
    row = np.array([0, 0, 1, 1])
    col = np.array([0, 1, 2, 3])
    data = np.array([4, 5, 7, 9])
    test_arr = scipy.sparse.coo_array((data, (row, col)), shape=(2, 4))
    test_mat = scipy.sparse.coo_matrix((data, (row, col)), shape=(2, 4))
    assert_equal(spnorm(test_arr, ord=1, axis=0), np.array([4, 5, 7, 9]))
    assert_equal(spnorm(test_mat, ord=1, axis=0), np.array([4, 5, 7, 9]))
    assert_equal(spnorm(test_arr, ord=1, axis=1), np.array([9, 16]))
    assert_equal(spnorm(test_mat, ord=1, axis=1), np.array([9, 16]))