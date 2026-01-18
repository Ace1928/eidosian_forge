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
def test_tocoo_gh10050(self):
    m = dia_matrix([[1, 2], [3, 4]]).tocoo()
    flat_inds = np.ravel_multi_index((m.row, m.col), m.shape)
    inds_are_sorted = np.all(np.diff(flat_inds) > 0)
    assert m.has_canonical_format == inds_are_sorted