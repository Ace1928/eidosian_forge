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
def test_self_self_assignment(self):
    B = self.spcreator((4, 3))
    with suppress_warnings() as sup:
        sup.filter(SparseEfficiencyWarning, 'Changing the sparsity structure of a cs[cr]_matrix is expensive')
        B[0, 0] = 2
        B[1, 2] = 7
        B[2, 1] = 3
        B[3, 0] = 10
        A = B / 10
        B[0, :] = A[0, :]
        assert_array_equal(A[0, :].toarray(), B[0, :].toarray())
        A = B / 10
        B[:, :] = A[:1, :1]
        assert_array_equal(np.zeros((4, 3)) + A[0, 0], B.toarray())
        A = B / 10
        B[:-1, 0] = A[0, :].T
        assert_array_equal(A[0, :].toarray().T, B[:-1, 0].toarray())