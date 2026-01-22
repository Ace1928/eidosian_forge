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
class TestCOONonCanonical(_NonCanonicalMixin, TestCOO):

    def _arg1_for_noncanonical(self, M, sorted_indices=None):
        """Return non-canonical constructor arg1 equivalent to M"""
        data, row, col = _same_sum_duplicate(M.data, M.row, M.col)
        return (data, (row, col))

    def _insert_explicit_zero(self, M, i, j):
        M.data = np.r_[M.data.dtype.type(0), M.data]
        M.row = np.r_[M.row.dtype.type(i), M.row]
        M.col = np.r_[M.col.dtype.type(j), M.col]
        return M

    def test_setdiag_noncanonical(self):
        m = self.spcreator(np.eye(3))
        m.sum_duplicates()
        m.setdiag([3, 2], k=1)
        m.sum_duplicates()
        assert_(np.all(np.diff(m.col) >= 0))