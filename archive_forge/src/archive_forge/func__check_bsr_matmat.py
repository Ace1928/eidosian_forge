import sys
import os
import gc
import threading
import numpy as np
from numpy.testing import assert_equal, assert_, assert_allclose
from scipy.sparse import (_sparsetools, coo_matrix, csr_matrix, csc_matrix,
from scipy.sparse._sputils import supported_dtypes
from scipy._lib._testutils import check_free_memory
import pytest
from pytest import raises as assert_raises
def _check_bsr_matmat(self, m):
    m = m()
    n = self.n
    m2 = bsr_matrix(np.ones((n, 2), dtype=np.int8), blocksize=(m.blocksize[1], 2))
    m.dot(m2)
    del m2
    m2 = bsr_matrix(np.ones((2, n), dtype=np.int8), blocksize=(2, m.blocksize[0]))
    m2.dot(m)