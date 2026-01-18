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
def test_sparse_format_conversions(self):
    A = sparse.kron([[1, 0, 2], [0, 3, 4], [5, 0, 0]], [[1, 2], [0, 3]])
    D = A.toarray()
    A = self.spcreator(A)
    for format in ['bsr', 'coo', 'csc', 'csr', 'dia', 'dok', 'lil']:
        a = A.asformat(format)
        assert_equal(a.format, format)
        assert_array_equal(a.toarray(), D)
        b = self.spcreator(D + 3j).asformat(format)
        assert_equal(b.format, format)
        assert_array_equal(b.toarray(), D + 3j)
        c = eval(format + '_matrix')(A)
        assert_equal(c.format, format)
        assert_array_equal(c.toarray(), D)
    for format in ['array', 'dense']:
        a = A.asformat(format)
        assert_array_equal(a, D)
        b = self.spcreator(D + 3j).asformat(format)
        assert_array_equal(b, D + 3j)