import pytest
import numpy as np
from numpy.testing import assert_allclose
import scipy.special as sc
class TestExp1:

    def test_branch_cut(self):
        assert np.isnan(sc.exp1(-1))
        assert sc.exp1(complex(-1, 0)).imag == -sc.exp1(complex(-1, -0.0)).imag
        assert_allclose(sc.exp1(complex(-1, 0)), sc.exp1(-1 + 1e-20j), atol=0, rtol=1e-15)
        assert_allclose(sc.exp1(complex(-1, -0.0)), sc.exp1(-1 - 1e-20j), atol=0, rtol=1e-15)

    def test_834(self):
        a = sc.exp1(-complex(19.999999))
        b = sc.exp1(-complex(19.9999991))
        assert_allclose(a.imag, b.imag, atol=0, rtol=1e-15)