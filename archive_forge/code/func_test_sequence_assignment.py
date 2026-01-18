import contextlib
import functools
import operator
import platform
import itertools
import sys
from scipy._lib import _pep440
import numpy as np
from numpy import (arange, zeros, array, dot, asarray,
import random
from numpy.testing import (assert_equal, assert_array_equal,
from pytest import raises as assert_raises
import scipy.linalg
import scipy.sparse as sparse
from scipy.sparse import (csc_matrix, csr_matrix, dok_matrix,
from scipy.sparse._sputils import (supported_dtypes, isscalarlike,
from scipy.sparse.linalg import splu, expm, inv
from scipy._lib.decorator import decorator
from scipy._lib._util import ComplexWarning
import pytest
def test_sequence_assignment(self):
    A = self.spcreator((4, 3))
    B = self.spcreator(eye(3, 4))
    i0 = [0, 1, 2]
    i1 = (0, 1, 2)
    i2 = array(i0)
    with suppress_warnings() as sup:
        sup.filter(SparseEfficiencyWarning, 'Changing the sparsity structure of a cs[cr]_matrix is expensive')
        with check_remains_sorted(A):
            A[0, i0] = B[i0, 0].T
            A[1, i1] = B[i1, 1].T
            A[2, i2] = B[i2, 2].T
        assert_array_equal(A.toarray(), B.T.toarray())
        A = self.spcreator((2, 3))
        with check_remains_sorted(A):
            A[1, 1:3] = [10, 20]
        assert_array_equal(A.toarray(), [[0, 0, 0], [0, 10, 20]])
        A = self.spcreator((3, 2))
        with check_remains_sorted(A):
            A[1:3, 1] = [[10], [20]]
        assert_array_equal(A.toarray(), [[0, 0], [0, 10], [0, 20]])
        A = self.spcreator((3, 3))
        B = asmatrix(np.zeros((3, 3)))
        with check_remains_sorted(A):
            for C in [A, B]:
                C[[0, 1, 2], [0, 1, 2]] = [4, 5, 6]
        assert_array_equal(A.toarray(), B)
        A = self.spcreator((4, 3))
        with check_remains_sorted(A):
            A[(1, 2, 3), (0, 1, 2)] = [1, 2, 3]
        assert_almost_equal(A.sum(), 6)
        B = asmatrix(np.zeros((4, 3)))
        B[(1, 2, 3), (0, 1, 2)] = [1, 2, 3]
        assert_array_equal(A.toarray(), B)