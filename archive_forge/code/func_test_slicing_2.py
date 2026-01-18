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
def test_slicing_2(self):
    B = asmatrix(arange(50).reshape(5, 10))
    A = self.spcreator(B)
    assert_equal(A[2, 3], B[2, 3])
    assert_equal(A[-1, 8], B[-1, 8])
    assert_equal(A[-1, -2], B[-1, -2])
    assert_equal(A[array(-1), -2], B[-1, -2])
    assert_equal(A[-1, array(-2)], B[-1, -2])
    assert_equal(A[array(-1), array(-2)], B[-1, -2])
    assert_equal(A[2, :].toarray(), B[2, :])
    assert_equal(A[2, 5:-2].toarray(), B[2, 5:-2])
    assert_equal(A[array(2), 5:-2].toarray(), B[2, 5:-2])
    assert_equal(A[:, 2].toarray(), B[:, 2])
    assert_equal(A[3:4, 9].toarray(), B[3:4, 9])
    assert_equal(A[1:4, -5].toarray(), B[1:4, -5])
    assert_equal(A[2:-1, 3].toarray(), B[2:-1, 3])
    assert_equal(A[2:-1, array(3)].toarray(), B[2:-1, 3])
    assert_equal(A[1:2, 1:2].toarray(), B[1:2, 1:2])
    assert_equal(A[4:, 3:].toarray(), B[4:, 3:])
    assert_equal(A[:4, :5].toarray(), B[:4, :5])
    assert_equal(A[2:-1, :5].toarray(), B[2:-1, :5])
    assert_equal(A[1, :].toarray(), B[1, :])
    assert_equal(A[-2, :].toarray(), B[-2, :])
    assert_equal(A[array(-2), :].toarray(), B[-2, :])
    assert_equal(A[1:4].toarray(), B[1:4])
    assert_equal(A[1:-2].toarray(), B[1:-2])
    s = slice(int8(2), int8(4), None)
    assert_equal(A[s, :].toarray(), B[2:4, :])
    assert_equal(A[:, s].toarray(), B[:, 2:4])