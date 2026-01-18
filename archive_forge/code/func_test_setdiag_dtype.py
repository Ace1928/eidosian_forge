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
@with_64bit_maxval_limit(3)
def test_setdiag_dtype(self):
    m = dia_matrix(np.eye(3))
    assert_equal(m.offsets.dtype, np.int32)
    m.setdiag((3,), k=2)
    assert_equal(m.offsets.dtype, np.int32)
    m = dia_matrix(np.eye(4))
    assert_equal(m.offsets.dtype, np.int64)
    m.setdiag((3,), k=3)
    assert_equal(m.offsets.dtype, np.int64)