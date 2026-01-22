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
class TestBSRNonCanonical(_NonCanonicalCompressedMixin, TestBSR):

    def _insert_explicit_zero(self, M, i, j):
        x = M.tocsr()
        x[i, j] = 0
        return x.tobsr(blocksize=M.blocksize)

    @pytest.mark.xfail(run=False, reason='diagonal broken with non-canonical BSR')
    def test_diagonal(self):
        pass

    @pytest.mark.xfail(run=False, reason='expm broken with non-canonical BSR')
    def test_expm(self):
        pass