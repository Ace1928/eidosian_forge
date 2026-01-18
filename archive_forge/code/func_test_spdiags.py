import numpy as np
from numpy import array
from numpy.testing import (assert_equal, assert_,
import pytest
from pytest import raises as assert_raises
from scipy._lib._testutils import check_free_memory
from scipy._lib._util import check_random_state
from scipy.sparse import (csr_matrix, coo_matrix,
from scipy.sparse._construct import rand as sprand
def test_spdiags(self):
    diags1 = array([[1, 2, 3, 4, 5]])
    diags2 = array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    diags3 = array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
    cases = []
    cases.append((diags1, 0, 1, 1, [[1]]))
    cases.append((diags1, [0], 1, 1, [[1]]))
    cases.append((diags1, [0], 2, 1, [[1], [0]]))
    cases.append((diags1, [0], 1, 2, [[1, 0]]))
    cases.append((diags1, [1], 1, 2, [[0, 2]]))
    cases.append((diags1, [-1], 1, 2, [[0, 0]]))
    cases.append((diags1, [0], 2, 2, [[1, 0], [0, 2]]))
    cases.append((diags1, [-1], 2, 2, [[0, 0], [1, 0]]))
    cases.append((diags1, [3], 2, 2, [[0, 0], [0, 0]]))
    cases.append((diags1, [0], 3, 4, [[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0]]))
    cases.append((diags1, [1], 3, 4, [[0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]]))
    cases.append((diags1, [2], 3, 5, [[0, 0, 3, 0, 0], [0, 0, 0, 4, 0], [0, 0, 0, 0, 5]]))
    cases.append((diags2, [0, 2], 3, 3, [[1, 0, 8], [0, 2, 0], [0, 0, 3]]))
    cases.append((diags2, [-1, 0], 3, 4, [[6, 0, 0, 0], [1, 7, 0, 0], [0, 2, 8, 0]]))
    cases.append((diags2, [2, -3], 6, 6, [[0, 0, 3, 0, 0, 0], [0, 0, 0, 4, 0, 0], [0, 0, 0, 0, 5, 0], [6, 0, 0, 0, 0, 0], [0, 7, 0, 0, 0, 0], [0, 0, 8, 0, 0, 0]]))
    cases.append((diags3, [-1, 0, 1], 6, 6, [[6, 12, 0, 0, 0, 0], [1, 7, 13, 0, 0, 0], [0, 2, 8, 14, 0, 0], [0, 0, 3, 9, 15, 0], [0, 0, 0, 4, 10, 0], [0, 0, 0, 0, 5, 0]]))
    cases.append((diags3, [-4, 2, -1], 6, 5, [[0, 0, 8, 0, 0], [11, 0, 0, 9, 0], [0, 12, 0, 0, 10], [0, 0, 13, 0, 0], [1, 0, 0, 14, 0], [0, 2, 0, 0, 15]]))
    cases.append((diags3, [-1, 1, 2], len(diags3[0]), len(diags3[0]), [[0, 7, 13, 0, 0], [1, 0, 8, 14, 0], [0, 2, 0, 9, 15], [0, 0, 3, 0, 10], [0, 0, 0, 4, 0]]))
    for d, o, m, n, result in cases:
        if len(d[0]) == m and m == n:
            assert_equal(construct.spdiags(d, o).toarray(), result)
        assert_equal(construct.spdiags(d, o, m, n).toarray(), result)
        assert_equal(construct.spdiags(d, o, (m, n)).toarray(), result)