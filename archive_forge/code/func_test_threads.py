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
def test_threads():
    nthreads = 10
    niter = 100
    n = 20
    a = csr_matrix(np.ones([n, n]))
    bres = []

    class Worker(threading.Thread):

        def run(self):
            b = a.copy()
            for j in range(niter):
                _sparsetools.csr_plus_csr(n, n, a.indptr, a.indices, a.data, a.indptr, a.indices, a.data, b.indptr, b.indices, b.data)
            bres.append(b)
    threads = [Worker() for _ in range(nthreads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for b in bres:
        assert_(np.all(b.toarray() == 2))