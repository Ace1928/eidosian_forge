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
class _TestGetNnzAxis:

    def test_getnnz_axis(self):
        dat = array([[0, 2], [3, 5], [-6, 9]])
        bool_dat = dat.astype(bool)
        datsp = self.spcreator(dat)
        accepted_return_dtypes = (np.int32, np.int64)
        assert_array_equal(bool_dat.sum(axis=None), datsp.getnnz(axis=None))
        assert_array_equal(bool_dat.sum(), datsp.getnnz())
        assert_array_equal(bool_dat.sum(axis=0), datsp.getnnz(axis=0))
        assert_in(datsp.getnnz(axis=0).dtype, accepted_return_dtypes)
        assert_array_equal(bool_dat.sum(axis=1), datsp.getnnz(axis=1))
        assert_in(datsp.getnnz(axis=1).dtype, accepted_return_dtypes)
        assert_array_equal(bool_dat.sum(axis=-2), datsp.getnnz(axis=-2))
        assert_in(datsp.getnnz(axis=-2).dtype, accepted_return_dtypes)
        assert_array_equal(bool_dat.sum(axis=-1), datsp.getnnz(axis=-1))
        assert_in(datsp.getnnz(axis=-1).dtype, accepted_return_dtypes)
        assert_raises(ValueError, datsp.getnnz, axis=2)