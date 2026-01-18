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
def test_set_slice(self):
    A = self.spcreator((5, 10))
    B = array(zeros((5, 10), float))
    s_ = np.s_
    slices = [s_[:2], s_[1:2], s_[3:], s_[3::2], s_[8:3:-1], s_[4::-2], s_[:5:-1], 0, 1, s_[:], s_[1:5], -1, -2, -5, array(-1), np.int8(-3)]
    with suppress_warnings() as sup:
        sup.filter(SparseEfficiencyWarning, 'Changing the sparsity structure of a cs[cr]_matrix is expensive')
        for j, a in enumerate(slices):
            A[a] = j
            B[a] = j
            assert_array_equal(A.toarray(), B, repr(a))
        for i, a in enumerate(slices):
            for j, b in enumerate(slices):
                A[a, b] = 10 * i + 1000 * (j + 1)
                B[a, b] = 10 * i + 1000 * (j + 1)
                assert_array_equal(A.toarray(), B, repr((a, b)))
        A[0, 1:10:2] = range(1, 10, 2)
        B[0, 1:10:2] = range(1, 10, 2)
        assert_array_equal(A.toarray(), B)
        A[1:5:2, 0] = np.arange(1, 5, 2)[:, None]
        B[1:5:2, 0] = np.arange(1, 5, 2)[:]
        assert_array_equal(A.toarray(), B)
    assert_raises(ValueError, A.__setitem__, (0, 0), list(range(100)))
    assert_raises(ValueError, A.__setitem__, (0, 0), arange(100))
    assert_raises(ValueError, A.__setitem__, (0, slice(None)), list(range(100)))
    assert_raises(ValueError, A.__setitem__, (slice(None), 1), list(range(100)))
    assert_raises(ValueError, A.__setitem__, (slice(None), 1), A.copy())
    assert_raises(ValueError, A.__setitem__, ([[1, 2, 3], [0, 3, 4]], [1, 2, 3]), [1, 2, 3, 4])
    assert_raises(ValueError, A.__setitem__, ([[1, 2, 3], [0, 3, 4], [4, 1, 3]], [[1, 2, 4], [0, 1, 3]]), [2, 3, 4])
    assert_raises(ValueError, A.__setitem__, (slice(4), 0), [[1, 2], [3, 4]])