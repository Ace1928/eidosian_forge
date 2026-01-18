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
def test_tocoo_tocsr_tocsc_gh19245(self):
    data = np.array([[1, 2, 3, 4]]).repeat(3, axis=0)
    offsets = np.array([0, -1, 2], dtype=np.int32)
    dia = sparse.dia_array((data, offsets), shape=(4, 4))
    coo = dia.tocoo()
    assert coo.col.dtype == np.int32
    csr = dia.tocsr()
    assert csr.indices.dtype == np.int32
    csc = dia.tocsc()
    assert csc.indices.dtype == np.int32