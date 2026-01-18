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
@pytest.mark.skipif(IS_COLAB, reason='exceeds memory limit')
def test_scalar_idx_dtype(self):
    indptr = np.zeros(2, dtype=np.int32)
    indices = np.zeros(0, dtype=np.int32)
    vals = np.zeros((0, 1, 1))
    a = bsr_matrix((vals, indices, indptr), shape=(1, 2 ** 31 - 1))
    b = bsr_matrix((vals, indices, indptr), shape=(1, 2 ** 31))
    c = bsr_matrix((1, 2 ** 31 - 1))
    d = bsr_matrix((1, 2 ** 31))
    assert_equal(a.indptr.dtype, np.int32)
    assert_equal(b.indptr.dtype, np.int64)
    assert_equal(c.indptr.dtype, np.int32)
    assert_equal(d.indptr.dtype, np.int64)
    try:
        vals2 = np.zeros((0, 1, 2 ** 31 - 1))
        vals3 = np.zeros((0, 1, 2 ** 31))
        e = bsr_matrix((vals2, indices, indptr), shape=(1, 2 ** 31 - 1))
        f = bsr_matrix((vals3, indices, indptr), shape=(1, 2 ** 31))
        assert_equal(e.indptr.dtype, np.int32)
        assert_equal(f.indptr.dtype, np.int64)
    except (MemoryError, ValueError):
        e = 0
        f = 0
    for x in [a, b, c, d, e, f]:
        x + x