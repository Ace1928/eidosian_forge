import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
import pytest
import scipy.linalg
import scipy.sparse.linalg
from scipy.sparse.linalg._onenormest import _onenormest_core, _algorithm_2_2
def _help_product_norm_slow(self, A, B):
    C = np.dot(A, B)
    return scipy.linalg.norm(C, 1)