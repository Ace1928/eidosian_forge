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
def test_sort_indices(self):
    data = arange(5)
    row = array([7, 2, 1, 5, 4])
    ptr = [0, 3, 5]
    asp = csc_matrix((data, row, ptr), shape=(10, 2))
    bsp = asp.copy()
    asp.sort_indices()
    assert_array_equal(asp.indices, [1, 2, 7, 4, 5])
    assert_array_equal(asp.toarray(), bsp.toarray())