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
class _TestArithmetic:
    """
    Test real/complex arithmetic
    """

    def __arith_init(self):
        self.__A = array([[-1.5, 6.5, 0, 2.25, 0, 0], [3.125, -7.875, 0.625, 0, 0, 0], [0, 0, -0.125, 1.0, 0, 0], [0, 0, 8.375, 0, 0, 0]], 'float64')
        self.__B = array([[0.375, 0, 0, 0, -5, 2.5], [14.25, -3.75, 0, 0, -0.125, 0], [0, 7.25, 0, 0, 0, 0], [18.5, -0.0625, 0, 0, 0, 0]], 'complex128')
        self.__B.imag = array([[1.25, 0, 0, 0, 6, -3.875], [2.25, 4.125, 0, 0, 0, 2.75], [0, 4.125, 0, 0, 0, 0], [-0.0625, 0, 0, 0, 0, 0]], 'float64')
        assert_array_equal((self.__A * 16).astype('int32'), 16 * self.__A)
        assert_array_equal((self.__B.real * 16).astype('int32'), 16 * self.__B.real)
        assert_array_equal((self.__B.imag * 16).astype('int32'), 16 * self.__B.imag)
        self.__Asp = self.spcreator(self.__A)
        self.__Bsp = self.spcreator(self.__B)

    def test_add_sub(self):
        self.__arith_init()
        assert_array_equal((self.__Asp + self.__Bsp).toarray(), self.__A + self.__B)
        for x in supported_dtypes:
            with np.errstate(invalid='ignore'):
                A = self.__A.astype(x)
            Asp = self.spcreator(A)
            for y in supported_dtypes:
                if not np.issubdtype(y, np.complexfloating):
                    with np.errstate(invalid='ignore'):
                        B = self.__B.real.astype(y)
                else:
                    B = self.__B.astype(y)
                Bsp = self.spcreator(B)
                D1 = A + B
                S1 = Asp + Bsp
                assert_equal(S1.dtype, D1.dtype)
                assert_array_equal(S1.toarray(), D1)
                assert_array_equal(Asp + B, D1)
                assert_array_equal(A + Bsp, D1)
                if np.dtype('bool') in [x, y]:
                    continue
                D1 = A - B
                S1 = Asp - Bsp
                assert_equal(S1.dtype, D1.dtype)
                assert_array_equal(S1.toarray(), D1)
                assert_array_equal(Asp - B, D1)
                assert_array_equal(A - Bsp, D1)

    def test_mu(self):
        self.__arith_init()
        assert_array_equal((self.__Asp @ self.__Bsp.T).toarray(), self.__A @ self.__B.T)
        for x in supported_dtypes:
            with np.errstate(invalid='ignore'):
                A = self.__A.astype(x)
            Asp = self.spcreator(A)
            for y in supported_dtypes:
                if np.issubdtype(y, np.complexfloating):
                    B = self.__B.astype(y)
                else:
                    with np.errstate(invalid='ignore'):
                        B = self.__B.real.astype(y)
                Bsp = self.spcreator(B)
                D1 = A @ B.T
                S1 = Asp @ Bsp.T
                assert_allclose(S1.toarray(), D1, atol=1e-14 * abs(D1).max())
                assert_equal(S1.dtype, D1.dtype)