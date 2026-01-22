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
class _TestFancyMultidimAssign:

    def test_fancy_assign_ndarray(self):
        np.random.seed(1234)
        D = asmatrix(np.random.rand(5, 7))
        S = self.spcreator(D)
        X = np.random.rand(2, 3)
        I = np.array([[1, 2, 3], [3, 4, 2]])
        J = np.array([[5, 6, 3], [2, 3, 1]])
        with check_remains_sorted(S):
            S[I, J] = X
        D[I, J] = X
        assert_equal(S.toarray(), D)
        I_bad = I + 5
        J_bad = J + 7
        C = [1, 2, 3]
        with check_remains_sorted(S):
            S[I, J] = C
        D[I, J] = C
        assert_equal(S.toarray(), D)
        with check_remains_sorted(S):
            S[I, J] = 3
        D[I, J] = 3
        assert_equal(S.toarray(), D)
        assert_raises(IndexError, S.__setitem__, (I_bad, J), C)
        assert_raises(IndexError, S.__setitem__, (I, J_bad), C)

    def test_fancy_indexing_multidim_set(self):
        n, m = (5, 10)

        def _test_set_slice(i, j):
            A = self.spcreator((n, m))
            with check_remains_sorted(A), suppress_warnings() as sup:
                sup.filter(SparseEfficiencyWarning, 'Changing the sparsity structure of a cs[cr]_matrix is expensive')
                A[i, j] = 1
            B = asmatrix(np.zeros((n, m)))
            B[i, j] = 1
            assert_array_almost_equal(A.toarray(), B)
        for i, j in [(np.array([[1, 2], [1, 3]]), [1, 3]), (np.array([0, 4]), [[0, 3], [1, 2]]), ([[1, 2, 3], [0, 2, 4]], [[0, 4, 3], [4, 1, 2]])]:
            _test_set_slice(i, j)

    def test_fancy_assign_list(self):
        np.random.seed(1234)
        D = asmatrix(np.random.rand(5, 7))
        S = self.spcreator(D)
        X = np.random.rand(2, 3)
        I = [[1, 2, 3], [3, 4, 2]]
        J = [[5, 6, 3], [2, 3, 1]]
        S[I, J] = X
        D[I, J] = X
        assert_equal(S.toarray(), D)
        I_bad = [[ii + 5 for ii in i] for i in I]
        J_bad = [[jj + 7 for jj in j] for j in J]
        C = [1, 2, 3]
        S[I, J] = C
        D[I, J] = C
        assert_equal(S.toarray(), D)
        S[I, J] = 3
        D[I, J] = 3
        assert_equal(S.toarray(), D)
        assert_raises(IndexError, S.__setitem__, (I_bad, J), C)
        assert_raises(IndexError, S.__setitem__, (I, J_bad), C)

    def test_fancy_assign_slice(self):
        np.random.seed(1234)
        D = asmatrix(np.random.rand(5, 7))
        S = self.spcreator(D)
        I = [1, 2, 3, 3, 4, 2]
        J = [5, 6, 3, 2, 3, 1]
        I_bad = [ii + 5 for ii in I]
        J_bad = [jj + 7 for jj in J]
        C1 = [1, 2, 3, 4, 5, 6, 7]
        C2 = np.arange(5)[:, None]
        assert_raises(IndexError, S.__setitem__, (I_bad, slice(None)), C1)
        assert_raises(IndexError, S.__setitem__, (slice(None), J_bad), C2)