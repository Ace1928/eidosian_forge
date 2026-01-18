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
def test_slicing_3(self):
    B = asmatrix(arange(50).reshape(5, 10))
    A = self.spcreator(B)
    s_ = np.s_
    slices = [s_[:2], s_[1:2], s_[3:], s_[3::2], s_[15:20], s_[3:2], s_[8:3:-1], s_[4::-2], s_[:5:-1], 0, 1, s_[:], s_[1:5], -1, -2, -5, array(-1), np.int8(-3)]

    def check_1(a):
        x = A[a]
        y = B[a]
        if y.shape == ():
            assert_equal(x, y, repr(a))
        elif x.size == 0 and y.size == 0:
            pass
        else:
            assert_array_equal(x.toarray(), y, repr(a))
    for j, a in enumerate(slices):
        check_1(a)

    def check_2(a, b):
        if isinstance(a, np.ndarray):
            ai = int(a)
        else:
            ai = a
        if isinstance(b, np.ndarray):
            bi = int(b)
        else:
            bi = b
        x = A[a, b]
        y = B[ai, bi]
        if y.shape == ():
            assert_equal(x, y, repr((a, b)))
        elif x.size == 0 and y.size == 0:
            pass
        else:
            assert_array_equal(x.toarray(), y, repr((a, b)))
    for i, a in enumerate(slices):
        for j, b in enumerate(slices):
            check_2(a, b)
    extra_slices = []
    for a, b, c in itertools.product(*[(None, 0, 1, 2, 5, 15, -1, -2, 5, -15)] * 3):
        if c == 0:
            continue
        extra_slices.append(slice(a, b, c))
    for a in extra_slices:
        check_2(a, a)
        check_2(a, -2)
        check_2(-2, a)