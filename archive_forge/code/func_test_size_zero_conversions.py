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
def test_size_zero_conversions(self):
    mat = array([])
    a = mat.reshape((0, 0))
    b = mat.reshape((0, 5))
    c = mat.reshape((5, 0))
    for m in [a, b, c]:
        spm = self.spcreator(m)
        assert_array_equal(spm.tocoo().toarray(), m)
        assert_array_equal(spm.tocsr().toarray(), m)
        assert_array_equal(spm.tocsc().toarray(), m)
        assert_array_equal(spm.tolil().toarray(), m)
        assert_array_equal(spm.todok().toarray(), m)
        assert_array_equal(spm.tobsr().toarray(), m)