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
def test_sub_dense(self):

    def check(dtype):
        dat = self.dat_dtypes[dtype]
        datsp = self.datsp_dtypes[dtype]
        if dat.dtype == bool:
            sum1 = dat - datsp
            assert_array_equal(sum1, dat - dat)
            sum2 = datsp - dat
            assert_array_equal(sum2, dat - dat)
        else:
            sum1 = dat + dat + dat - datsp
            assert_array_equal(sum1, dat + dat)
            sum2 = datsp + datsp + datsp - dat
            assert_array_equal(sum2, dat + dat)
    for dtype in self.math_dtypes:
        if dtype == np.dtype('bool'):
            continue
        check(dtype)