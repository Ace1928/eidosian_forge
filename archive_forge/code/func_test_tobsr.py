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
def test_tobsr(self):
    x = array([[1, 0, 2, 0], [0, 0, 0, 0], [0, 0, 4, 5]])
    y = array([[0, 1, 2], [3, 0, 5]])
    A = kron(x, y)
    Asp = self.spcreator(A)
    for format in ['bsr']:
        fn = getattr(Asp, 'to' + format)
        for X in [1, 2, 3, 6]:
            for Y in [1, 2, 3, 4, 6, 12]:
                assert_equal(fn(blocksize=(X, Y)).toarray(), A)