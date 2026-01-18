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
def test_start_vs_at_sign_for_sparray_and_spmatrix(self):
    A = self.spcreator([[1], [2], [3]])
    if isinstance(A, sparray):
        assert_array_almost_equal(A * np.ones((3, 1)), A)
        assert_array_almost_equal(A * array([[1]]), A)
        assert_array_almost_equal(A * np.ones((3, 1)), A)
    else:
        assert_equal(A * array([1]), array([1, 2, 3]))
        assert_equal(A * array([[1]]), array([[1], [2], [3]]))
        assert_equal(A * np.ones((1, 0)), np.ones((3, 0)))