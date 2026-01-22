import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import scipy.special as sc
class TestHyp1f1:

    @pytest.mark.parametrize('a, b, x', [(np.nan, 1, 1), (1, np.nan, 1), (1, 1, np.nan)])
    def test_nan_inputs(self, a, b, x):
        assert np.isnan(sc.hyp1f1(a, b, x))

    def test_poles(self):
        assert_equal(sc.hyp1f1(1, [0, -1, -2, -3, -4], 0.5), np.inf)

    @pytest.mark.parametrize('a, b, x, result', [(-1, 1, 0.5, 0.5), (1, 1, 0.5, 1.6487212707001282), (2, 1, 0.5, 2.4730819060501923), (1, 2, 0.5, 1.2974425414002564), (-10, 1, 0.5, -0.38937441413785207)])
    def test_special_cases(self, a, b, x, result):
        assert_allclose(sc.hyp1f1(a, b, x), result, atol=0, rtol=1e-15)

    @pytest.mark.parametrize('a, b, x, result', [(1, 1, 0.44, 1.552707218511336), (-1, 1, 0.44, 0.56), (100, 100, 0.89, 2.4351296512898744), (-100, 100, 0.89, 0.407390624907681), (1.5, 100, 59.99, 3.8073513625965596), (-1.5, 100, 59.99, 0.25099240047125826)])
    def test_geometric_convergence(self, a, b, x, result):
        assert_allclose(sc.hyp1f1(a, b, x), result, atol=0, rtol=1e-15)

    @pytest.mark.parametrize('a, b, x, result', [(-1, 1, 1.5, -0.5), (-10, 1, 1.5, 0.4180177743094308), (-25, 1, 1.5, 0.2511449164603784), (-50, 1, 1.5, -0.2568364397519476), (-80, 1, 1.5, -0.24554329325751503), (-150, 1, 1.5, -0.17336479551542044)])
    def test_a_negative_integer(self, a, b, x, result):
        assert_allclose(sc.hyp1f1(a, b, x), result, atol=0, rtol=2e-14)

    @pytest.mark.parametrize('a, b, x, expected', [(0.01, 150, -4, 0.9997368389767752), (1, 5, 0.01, 1.002003338101197), (50, 100, 0.01, 1.0050126452421464), (1, 0.3, -1000.0, -0.0007011932249442948), (1, 0.3, -10000.0, -7.001190321418937e-05), (9, 8.5, -350, -5.2240908319223784e-20), (9, 8.5, -355, -4.595407159813368e-20), (75, -123.5, 15, 3425753.920814889)])
    def test_assorted_cases(self, a, b, x, expected):
        assert_allclose(sc.hyp1f1(a, b, x), expected, atol=0, rtol=1e-14)

    def test_a_neg_int_and_b_equal_x(self):
        a = -10.0
        b = 2.5
        x = 2.5
        expected = 0.03653236643641043
        computed = sc.hyp1f1(a, b, x)
        assert_allclose(computed, expected, atol=0, rtol=1e-13)

    @pytest.mark.parametrize('a, b, x, desired', [(-1, -2, 2, 2), (-1, -4, 10, 3.5), (-2, -2, 1, 2.5)])
    def test_gh_11099(self, a, b, x, desired):
        assert sc.hyp1f1(a, b, x) == desired

    @pytest.mark.parametrize('a', [-3, -2])
    def test_x_zero_a_and_b_neg_ints_and_a_ge_b(self, a):
        assert sc.hyp1f1(a, -3, 0) == 1

    @pytest.mark.parametrize('b', [0, -1, -5])
    def test_legacy_case1(self, b):
        assert_equal(sc.hyp1f1(0, b, [-1.5, 0, 1.5]), [np.inf, np.inf, np.inf])

    def test_legacy_case2(self):
        assert sc.hyp1f1(-4, -3, 0) == np.inf