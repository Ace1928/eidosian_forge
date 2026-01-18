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
def test_p_max_default(self):
    m_max = 55
    expected_p_max = 8
    observed_p_max = _compute_p_max(m_max)
    assert_equal(observed_p_max, expected_p_max)