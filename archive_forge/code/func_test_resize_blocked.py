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
def test_resize_blocked(self):
    D = np.array([[1, 0, 3, 4], [2, 0, 0, 0], [3, 0, 0, 0]])
    S = self.spcreator(D, blocksize=(1, 2))
    assert_(S.resize((3, 2)) is None)
    assert_array_equal(S.toarray(), [[1, 0], [2, 0], [3, 0]])
    S.resize((2, 2))
    assert_array_equal(S.toarray(), [[1, 0], [2, 0]])
    S.resize((3, 2))
    assert_array_equal(S.toarray(), [[1, 0], [2, 0], [0, 0]])
    S.resize((3, 4))
    assert_array_equal(S.toarray(), [[1, 0, 0, 0], [2, 0, 0, 0], [0, 0, 0, 0]])
    assert_raises(ValueError, S.resize, (2, 3))