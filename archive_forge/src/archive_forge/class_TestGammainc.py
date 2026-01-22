import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import scipy.special as sc
from scipy.special._testutils import FuncData
class TestGammainc:

    @pytest.mark.parametrize('a, x', INVALID_POINTS)
    def test_domain(self, a, x):
        assert np.isnan(sc.gammainc(a, x))

    def test_a_eq_0_x_gt_0(self):
        assert sc.gammainc(0, 1) == 1

    @pytest.mark.parametrize('a, x, desired', [(np.inf, 1, 0), (np.inf, 0, 0), (np.inf, np.inf, np.nan), (1, np.inf, 1)])
    def test_infinite_arguments(self, a, x, desired):
        result = sc.gammainc(a, x)
        if np.isnan(desired):
            assert np.isnan(result)
        else:
            assert result == desired

    def test_infinite_limits(self):
        assert_allclose(sc.gammainc(1000, 100), sc.gammainc(np.inf, 100), atol=1e-200, rtol=0)
        assert sc.gammainc(100, 1000) == sc.gammainc(100, np.inf)

    def test_x_zero(self):
        a = np.arange(1, 10)
        assert_array_equal(sc.gammainc(a, 0), 0)

    def test_limit_check(self):
        result = sc.gammainc(1e-10, 1)
        limit = sc.gammainc(0, 1)
        assert np.isclose(result, limit)

    def gammainc_line(self, x):
        c = np.array([-1 / 3, -1 / 540, 25 / 6048, 101 / 155520, -3184811 / 3695155200, -2745493 / 8151736420])
        res = 0
        xfac = 1
        for ck in c:
            res -= ck * xfac
            xfac /= x
        res /= np.sqrt(2 * np.pi * x)
        res += 0.5
        return res

    def test_line(self):
        x = np.logspace(np.log10(25), 300, 500)
        a = x
        dataset = np.vstack((a, x, self.gammainc_line(x))).T
        FuncData(sc.gammainc, dataset, (0, 1), 2, rtol=1e-11).check()

    def test_roundtrip(self):
        a = np.logspace(-5, 10, 100)
        x = np.logspace(-5, 10, 100)
        y = sc.gammaincinv(a, sc.gammainc(a, x))
        assert_allclose(x, y, rtol=1e-10)