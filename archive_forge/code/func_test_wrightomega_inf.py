import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_allclose
import scipy.special as sc
from scipy.special._testutils import assert_func_equal
def test_wrightomega_inf():
    pts = [complex(np.inf, 10), complex(-np.inf, 10), complex(10, np.inf), complex(10, -np.inf)]
    for p in pts:
        assert_equal(sc.wrightomega(p), p)