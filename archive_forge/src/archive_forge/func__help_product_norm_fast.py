import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
import pytest
import scipy.linalg
import scipy.sparse.linalg
from scipy.sparse.linalg._onenormest import _onenormest_core, _algorithm_2_2
def _help_product_norm_fast(self, A, B):
    t = 2
    itmax = 5
    D = MatrixProductOperator(A, B)
    est, v, w, nmults, nresamples = _onenormest_core(D, D.T, t, itmax)
    return est