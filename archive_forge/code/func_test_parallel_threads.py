import threading
import itertools
import numpy as np
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from pytest import raises as assert_raises
import pytest
from numpy import dot, conj, random
from scipy.linalg import eig, eigh
from scipy.sparse import csc_matrix, csr_matrix, diags, rand
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg._eigen.arpack import (eigs, eigsh, arpack,
from scipy._lib._gcutils import assert_deallocated, IS_PYPY
def test_parallel_threads():
    results = []
    v0 = np.random.rand(50)

    def worker():
        x = diags([1, -2, 1], [-1, 0, 1], shape=(50, 50))
        w, v = eigs(x, k=3, v0=v0)
        results.append(w)
        w, v = eigsh(x, k=3, v0=v0)
        results.append(w)
    threads = [threading.Thread(target=worker) for k in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    worker()
    for r in results:
        assert_allclose(r, results[-1])