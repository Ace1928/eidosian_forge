from functools import partial
from itertools import product
import numpy as np
import pytest
from numpy.testing import (assert_allclose, assert_, assert_equal,
from scipy.sparse import SparseEfficiencyWarning
from scipy.sparse.linalg import aslinearoperator
import scipy.linalg
from scipy.sparse.linalg import expm as sp_expm
from scipy.sparse.linalg._expm_multiply import (_theta, _compute_p_max,
from scipy._lib._util import np_long
def test_theta_monotonicity(self):
    pairs = sorted(_theta.items())
    for (m_a, theta_a), (m_b, theta_b) in zip(pairs[:-1], pairs[1:]):
        assert_(theta_a < theta_b)