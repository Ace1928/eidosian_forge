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
def test_rsub(self):

    def check(dtype):
        dat = self.dat_dtypes[dtype]
        datsp = self.datsp_dtypes[dtype]
        assert_array_equal(dat - datsp, [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        assert_array_equal(datsp - dat, [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        assert_array_equal((0 - datsp).toarray(), -dat)
        A = self.spcreator(matrix([[1, 0, 0, 4], [-1, 0, 0, 0], [0, 8, 0, -5]], 'd'))
        assert_array_equal(dat - A, dat - A.toarray())
        assert_array_equal(A - dat, A.toarray() - dat)
        assert_array_equal(A.toarray() - datsp, A.toarray() - dat)
        assert_array_equal(datsp - A.toarray(), dat - A.toarray())
        assert_array_equal(dat[0] - datsp, dat[0] - dat)
    for dtype in self.math_dtypes:
        if dtype == np.dtype('bool'):
            continue
        check(dtype)