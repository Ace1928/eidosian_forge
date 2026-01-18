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
def test_sum_dtype(self):
    dat = array([[0, 1, 2], [3, -4, 5], [-6, 7, 9]])
    datsp = self.spcreator(dat)

    def check(dtype):
        dat_mean = dat.mean(dtype=dtype)
        datsp_mean = datsp.mean(dtype=dtype)
        assert_array_almost_equal(dat_mean, datsp_mean)
        assert_equal(dat_mean.dtype, datsp_mean.dtype)
    for dtype in self.checked_dtypes:
        check(dtype)