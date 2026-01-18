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
def test_regression_std_vector_dtypes():
    for dtype in supported_dtypes:
        ad = np.array([[1, 2], [3, 4]]).astype(dtype)
        a = csr_matrix(ad, dtype=dtype)
        assert_equal(a.getcol(0).toarray(), ad[:, :1])