import pytest
import numpy as np
from numpy.testing import assert_, assert_equal, assert_allclose
import scipy.special as sc
from scipy.special._testutils import assert_func_equal
def test_wrightomega_inf_branch():
    pts = [complex(-np.inf, np.pi / 4), complex(-np.inf, -np.pi / 4), complex(-np.inf, 3 * np.pi / 4), complex(-np.inf, -3 * np.pi / 4)]
    expected_results = [complex(0.0, 0.0), complex(0.0, -0.0), complex(-0.0, 0.0), complex(-0.0, -0.0)]
    for p, expected in zip(pts, expected_results):
        res = sc.wrightomega(p)
        assert_equal(res.real, expected.real)
        assert_equal(res.imag, expected.imag)