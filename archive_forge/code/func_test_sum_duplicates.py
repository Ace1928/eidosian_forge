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
def test_sum_duplicates(self):
    coo = coo_matrix((4, 3))
    coo.sum_duplicates()
    coo = coo_matrix(([1, 2], ([1, 0], [1, 0])))
    coo.sum_duplicates()
    assert_array_equal(coo.toarray(), [[2, 0], [0, 1]])
    coo = coo_matrix(([1, 2], ([1, 1], [1, 1])))
    coo.sum_duplicates()
    assert_array_equal(coo.toarray(), [[0, 0], [0, 3]])
    assert_array_equal(coo.row, [1])
    assert_array_equal(coo.col, [1])
    assert_array_equal(coo.data, [3])