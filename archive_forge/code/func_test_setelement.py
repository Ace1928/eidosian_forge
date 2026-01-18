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
def test_setelement(self):

    def check(dtype):
        A = self.spcreator((3, 4), dtype=dtype)
        with suppress_warnings() as sup:
            sup.filter(SparseEfficiencyWarning, 'Changing the sparsity structure of a cs[cr]_matrix is expensive')
            A[0, 0] = dtype.type(0)
            A[1, 2] = dtype.type(4.0)
            A[0, 1] = dtype.type(3)
            A[2, 0] = dtype.type(2.0)
            A[0, -1] = dtype.type(8)
            A[-1, -2] = dtype.type(7)
            A[0, 1] = dtype.type(5)
        if dtype != np.bool_:
            assert_array_equal(A.toarray(), [[0, 5, 0, 8], [0, 0, 4, 0], [2, 0, 7, 0]])
        for ij in [(0, 4), (-1, 4), (3, 0), (3, 4), (3, -1)]:
            assert_raises(IndexError, A.__setitem__, ij, 123.0)
        for v in [[1, 2, 3], array([1, 2, 3])]:
            assert_raises(ValueError, A.__setitem__, (0, 0), v)
        if not np.issubdtype(dtype, np.complexfloating) and dtype != np.bool_:
            for v in [3j]:
                assert_raises(TypeError, A.__setitem__, (0, 0), v)
    for dtype in supported_dtypes:
        check(np.dtype(dtype))