import numpy as np
from numpy import array
from numpy.testing import (assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
from scipy._lib._testutils import check_free_memory
from scipy._lib._util import check_random_state
from scipy.sparse import (csr_matrix, coo_matrix,
from scipy.sparse._construct import rand as sprand
def test_random_sparse_matrix_returns_correct_number_of_non_zero_elements(self):
    sparse_matrix = construct.random(10, 10, density=0.1265)
    assert_equal(sparse_matrix.count_nonzero(), 13)
    sparse_array = construct.random_array((10, 10), density=0.1265)
    assert_equal(sparse_array.count_nonzero(), 13)
    assert isinstance(sparse_array, sparray)
    shape = (2 ** 33, 2 ** 33)
    sparse_array = construct.random_array(shape, density=2.7105e-17)
    assert_equal(sparse_array.count_nonzero(), 2000)