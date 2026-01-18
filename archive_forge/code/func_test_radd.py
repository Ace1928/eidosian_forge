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
def test_radd(self):

    def check(dtype):
        dat = self.dat_dtypes[dtype]
        datsp = self.datsp_dtypes[dtype]
        a = dat.copy()
        a[0, 2] = 2.0
        b = datsp
        c = a + b
        assert_array_equal(c, a + b.toarray())
    for dtype in self.math_dtypes:
        check(dtype)