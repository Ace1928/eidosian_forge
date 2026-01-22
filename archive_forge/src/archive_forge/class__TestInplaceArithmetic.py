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
class _TestInplaceArithmetic:

    def test_inplace_dense(self):
        a = np.ones((3, 4))
        b = self.spcreator(a)
        x = a.copy()
        y = a.copy()
        x += a
        y += b
        assert_array_equal(x, y)
        x = a.copy()
        y = a.copy()
        x -= a
        y -= b
        assert_array_equal(x, y)
        x = a.copy()
        y = a.copy()
        if isinstance(b, sparray):
            assert_raises(ValueError, operator.imul, x, b.T)
            x = x * a
            y *= b
        else:
            assert_raises(ValueError, operator.imul, x, b)
            x = x.dot(a.T)
            y *= b.T
        assert_array_equal(x, y)
        assert_raises(TypeError, operator.ifloordiv, x, b)

    def test_imul_scalar(self):

        def check(dtype):
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]
            if np.can_cast(int, dtype, casting='same_kind'):
                a = datsp.copy()
                a *= 2
                b = dat.copy()
                b *= 2
                assert_array_equal(b, a.toarray())
            if np.can_cast(float, dtype, casting='same_kind'):
                a = datsp.copy()
                a *= 17.3
                b = dat.copy()
                b *= 17.3
                assert_array_equal(b, a.toarray())
        for dtype in self.math_dtypes:
            check(dtype)

    def test_idiv_scalar(self):

        def check(dtype):
            dat = self.dat_dtypes[dtype]
            datsp = self.datsp_dtypes[dtype]
            if np.can_cast(int, dtype, casting='same_kind'):
                a = datsp.copy()
                a /= 2
                b = dat.copy()
                b /= 2
                assert_array_equal(b, a.toarray())
            if np.can_cast(float, dtype, casting='same_kind'):
                a = datsp.copy()
                a /= 17.3
                b = dat.copy()
                b /= 17.3
                assert_array_equal(b, a.toarray())
        for dtype in self.math_dtypes:
            if not np.can_cast(dtype, np.dtype(int)):
                check(dtype)

    def test_inplace_success(self):
        a = self.spcreator(np.eye(5))
        b = self.spcreator(np.eye(5))
        bp = self.spcreator(np.eye(5))
        b += a
        bp = bp + a
        assert_allclose(b.toarray(), bp.toarray())
        b *= a
        bp = bp * a
        assert_allclose(b.toarray(), bp.toarray())
        b -= a
        bp = bp - a
        assert_allclose(b.toarray(), bp.toarray())
        assert_raises(TypeError, operator.ifloordiv, a, b)