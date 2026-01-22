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
def _compare_index_dtype(self, m, dtype):
    dtype = np.dtype(dtype)
    if isinstance(m, (csc_matrix, csr_matrix, bsr_matrix)):
        return m.indices.dtype == dtype and m.indptr.dtype == dtype
    elif isinstance(m, coo_matrix):
        return m.row.dtype == dtype and m.col.dtype == dtype
    elif isinstance(m, dia_matrix):
        return m.offsets.dtype == dtype
    else:
        raise ValueError(f'matrix {m!r} has no integer indices')