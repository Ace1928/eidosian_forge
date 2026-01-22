import functools
import itertools
import operator
import platform
import sys
import numpy as np
from numpy import (array, isnan, r_, arange, finfo, pi, sin, cos, tan, exp,
import pytest
from pytest import raises as assert_raises
from numpy.testing import (assert_equal, assert_almost_equal,
from scipy import special
import scipy.special._ufuncs as cephes
from scipy.special import ellipe, ellipk, ellipkm1
from scipy.special import elliprc, elliprd, elliprf, elliprg, elliprj
from scipy.special import mathieu_odd_coef, mathieu_even_coef, stirling2
from scipy._lib.deprecation import _NoValue
from scipy._lib._util import np_long, np_ulong
from scipy.special._basic import _FACTORIALK_LIMITS_64BITS, \
from scipy.special._testutils import with_special_errors, \
import math
class TestCephes:

    def test_airy(self):
        cephes.airy(0)

    def test_airye(self):
        cephes.airye(0)

    def test_binom(self):
        n = np.array([0.264, 4, 5.2, 17])
        k = np.array([2, 0.4, 7, 3.3])
        nk = np.array(np.broadcast_arrays(n[:, None], k[None, :])).reshape(2, -1).T
        rknown = np.array([[-0.097152, 0.9263051596159367, 0.01858423645695389, -0.007581020651518199], [6, 2.0214389119675666, 0, 2.9827344527963846], [10.92, 2.22993515861399, -0.00585728, 10.468891352063146], [136, 3.5252179590758828, 19448, 1024.5526916174495]])
        assert_func_equal(cephes.binom, rknown.ravel(), nk, rtol=1e-13)
        np.random.seed(1234)
        n = np.r_[np.arange(-7, 30), 1000 * np.random.rand(30) - 500]
        k = np.arange(0, 102)
        nk = np.array(np.broadcast_arrays(n[:, None], k[None, :])).reshape(2, -1).T
        assert_func_equal(cephes.binom, cephes.binom(nk[:, 0], nk[:, 1] * (1 + 1e-15)), nk, atol=1e-10, rtol=1e-10)

    def test_binom_2(self):
        np.random.seed(1234)
        n = np.r_[np.logspace(1, 300, 20)]
        k = np.arange(0, 102)
        nk = np.array(np.broadcast_arrays(n[:, None], k[None, :])).reshape(2, -1).T
        assert_func_equal(cephes.binom, cephes.binom(nk[:, 0], nk[:, 1] * (1 + 1e-15)), nk, atol=1e-10, rtol=1e-10)

    def test_binom_exact(self):

        @np.vectorize
        def binom_int(n, k):
            n = int(n)
            k = int(k)
            num = 1
            den = 1
            for i in range(1, k + 1):
                num *= i + n - k
                den *= i
            return float(num / den)
        np.random.seed(1234)
        n = np.arange(1, 15)
        k = np.arange(0, 15)
        nk = np.array(np.broadcast_arrays(n[:, None], k[None, :])).reshape(2, -1).T
        nk = nk[nk[:, 0] >= nk[:, 1]]
        assert_func_equal(cephes.binom, binom_int(nk[:, 0], nk[:, 1]), nk, atol=0, rtol=0)

    def test_binom_nooverflow_8346(self):
        dataset = [(1000, 500, 2.7028824094543655e+299), (1002, 501, 1.0800739688079122e+300), (1004, 502, 4.315992791690581e+300), (1006, 503, 1.7246810161626378e+301), (1008, 504, 6.891880092364192e+301), (1010, 505, 2.7540225794833545e+302), (1012, 506, 1.1005204853192376e+303), (1014, 507, 4.3977406375873285e+303), (1016, 508, 1.7573648610831252e+304), (1018, 509, 7.022554277884237e+304), (1020, 510, 2.8062677682996225e+305), (1022, 511, 1.1214087637706124e+306), (1024, 512, 4.481254552098971e+306), (1026, 513, 1.790754743041499e+307), (1028, 514, 7.156051054877897e+307)]
        dataset = np.asarray(dataset)
        FuncData(cephes.binom, dataset, (0, 1), 2, rtol=1e-12).check()

    def test_bdtr(self):
        assert_equal(cephes.bdtr(1, 1, 0.5), 1.0)

    def test_bdtri(self):
        assert_equal(cephes.bdtri(1, 3, 0.5), 0.5)

    def test_bdtrc(self):
        assert_equal(cephes.bdtrc(1, 3, 0.5), 0.5)

    def test_bdtrin(self):
        assert_equal(cephes.bdtrin(1, 0, 1), 5.0)

    def test_bdtrik(self):
        cephes.bdtrik(1, 3, 0.5)

    def test_bei(self):
        assert_equal(cephes.bei(0), 0.0)

    def test_beip(self):
        assert_equal(cephes.beip(0), 0.0)

    def test_ber(self):
        assert_equal(cephes.ber(0), 1.0)

    def test_berp(self):
        assert_equal(cephes.berp(0), 0.0)

    def test_besselpoly(self):
        assert_equal(cephes.besselpoly(0, 0, 0), 1.0)

    def test_btdtr(self):
        with pytest.deprecated_call(match='deprecated in SciPy 1.12.0'):
            y = special.btdtr(1, 1, 1)
        assert_equal(y, 1.0)

    def test_btdtri(self):
        with pytest.deprecated_call(match='deprecated in SciPy 1.12.0'):
            y = special.btdtri(1, 1, 1)
        assert_equal(y, 1.0)

    def test_btdtria(self):
        assert_equal(cephes.btdtria(1, 1, 1), 5.0)

    def test_btdtrib(self):
        assert_equal(cephes.btdtrib(1, 1, 1), 5.0)

    def test_cbrt(self):
        assert_approx_equal(cephes.cbrt(1), 1.0)

    def test_chdtr(self):
        assert_equal(cephes.chdtr(1, 0), 0.0)

    def test_chdtrc(self):
        assert_equal(cephes.chdtrc(1, 0), 1.0)

    def test_chdtri(self):
        assert_equal(cephes.chdtri(1, 1), 0.0)

    def test_chdtriv(self):
        assert_equal(cephes.chdtriv(0, 0), 5.0)

    def test_chndtr(self):
        assert_equal(cephes.chndtr(0, 1, 0), 0.0)
        values = np.array([[25.0, 20.0, 400, 4.12106551123962e-57], [25.0, 8.0, 250, 2.3988026526832426e-29], [0.001, 8.0, 40.0, 5.376180620136604e-24], [0.01, 8.0, 40.0, 5.453962310559995e-20], [20.0, 2.0, 107, 1.393907435558196e-09], [22.5, 2.0, 107, 7.118033071381059e-09], [25.0, 2.0, 107, 3.110412448298649e-08], [3.0, 2.0, 1.0, 0.6206436532195436], [350.0, 300.0, 10.0, 0.9388012800627641], [100.0, 13.5, 10.0, 0.9999999965010421], [700.0, 20.0, 400, 0.9999999992568065], [150.0, 13.5, 10.0, 0.9999999999999998], [160.0, 13.5, 10.0, 1.0]])
        cdf = cephes.chndtr(values[:, 0], values[:, 1], values[:, 2])
        assert_allclose(cdf, values[:, 3], rtol=1e-12)
        assert_almost_equal(cephes.chndtr(np.inf, np.inf, 0), 2.0)
        assert_almost_equal(cephes.chndtr(2, 1, np.inf), 0.0)
        assert_(np.isnan(cephes.chndtr(np.nan, 1, 2)))
        assert_(np.isnan(cephes.chndtr(5, np.nan, 2)))
        assert_(np.isnan(cephes.chndtr(5, 1, np.nan)))

    def test_chndtridf(self):
        assert_equal(cephes.chndtridf(0, 0, 1), 5.0)

    def test_chndtrinc(self):
        assert_equal(cephes.chndtrinc(0, 1, 0), 5.0)

    def test_chndtrix(self):
        assert_equal(cephes.chndtrix(0, 1, 0), 0.0)

    def test_cosdg(self):
        assert_equal(cephes.cosdg(0), 1.0)

    def test_cosm1(self):
        assert_equal(cephes.cosm1(0), 0.0)

    def test_cotdg(self):
        assert_almost_equal(cephes.cotdg(45), 1.0)

    def test_dawsn(self):
        assert_equal(cephes.dawsn(0), 0.0)
        assert_allclose(cephes.dawsn(1.23), 0.5005372774908177)

    def test_diric(self):
        n_odd = [1, 5, 25]
        x = np.array(2 * np.pi + 5e-05).astype(np.float32)
        assert_almost_equal(special.diric(x, n_odd), 1.0, decimal=7)
        x = np.array(2 * np.pi + 1e-09).astype(np.float64)
        assert_almost_equal(special.diric(x, n_odd), 1.0, decimal=15)
        x = np.array(2 * np.pi + 1e-15).astype(np.float64)
        assert_almost_equal(special.diric(x, n_odd), 1.0, decimal=15)
        if hasattr(np, 'float128'):
            x = np.array(2 * np.pi + 1e-12).astype(np.float128)
            assert_almost_equal(special.diric(x, n_odd), 1.0, decimal=19)
        n_even = [2, 4, 24]
        x = np.array(2 * np.pi + 1e-09).astype(np.float64)
        assert_almost_equal(special.diric(x, n_even), -1.0, decimal=15)
        x = np.arange(0.2 * np.pi, 1.0 * np.pi, 0.2 * np.pi)
        octave_result = [0.872677996249965, 0.539344662916632, 0.127322003750035, -0.206011329583298]
        assert_almost_equal(special.diric(x, 3), octave_result, decimal=15)

    def test_diric_broadcasting(self):
        x = np.arange(5)
        n = np.array([1, 3, 7])
        assert_(special.diric(x[:, np.newaxis], n).shape == (x.size, n.size))

    def test_ellipe(self):
        assert_equal(cephes.ellipe(1), 1.0)

    def test_ellipeinc(self):
        assert_equal(cephes.ellipeinc(0, 1), 0.0)

    def test_ellipj(self):
        cephes.ellipj(0, 1)

    def test_ellipk(self):
        assert_allclose(ellipk(0), pi / 2)

    def test_ellipkinc(self):
        assert_equal(cephes.ellipkinc(0, 0), 0.0)

    def test_erf(self):
        assert_equal(cephes.erf(0), 0.0)

    def test_erf_symmetry(self):
        x = 5.905732037710919
        assert_equal(cephes.erf(x) + cephes.erf(-x), 0.0)

    def test_erfc(self):
        assert_equal(cephes.erfc(0), 1.0)

    def test_exp10(self):
        assert_approx_equal(cephes.exp10(2), 100.0)

    def test_exp2(self):
        assert_equal(cephes.exp2(2), 4.0)

    def test_expm1(self):
        assert_equal(cephes.expm1(0), 0.0)
        assert_equal(cephes.expm1(np.inf), np.inf)
        assert_equal(cephes.expm1(-np.inf), -1)
        assert_equal(cephes.expm1(np.nan), np.nan)

    def test_expm1_complex(self):
        expm1 = cephes.expm1
        assert_equal(expm1(0 + 0j), 0 + 0j)
        assert_equal(expm1(complex(np.inf, 0)), complex(np.inf, 0))
        assert_equal(expm1(complex(np.inf, 1)), complex(np.inf, np.inf))
        assert_equal(expm1(complex(np.inf, 2)), complex(-np.inf, np.inf))
        assert_equal(expm1(complex(np.inf, 4)), complex(-np.inf, -np.inf))
        assert_equal(expm1(complex(np.inf, 5)), complex(np.inf, -np.inf))
        assert_equal(expm1(complex(1, np.inf)), complex(np.nan, np.nan))
        assert_equal(expm1(complex(0, np.inf)), complex(np.nan, np.nan))
        assert_equal(expm1(complex(np.inf, np.inf)), complex(np.inf, np.nan))
        assert_equal(expm1(complex(-np.inf, np.inf)), complex(-1, 0))
        assert_equal(expm1(complex(-np.inf, np.nan)), complex(-1, 0))
        assert_equal(expm1(complex(np.inf, np.nan)), complex(np.inf, np.nan))
        assert_equal(expm1(complex(0, np.nan)), complex(np.nan, np.nan))
        assert_equal(expm1(complex(1, np.nan)), complex(np.nan, np.nan))
        assert_equal(expm1(complex(np.nan, 1)), complex(np.nan, np.nan))
        assert_equal(expm1(complex(np.nan, np.nan)), complex(np.nan, np.nan))

    @pytest.mark.xfail(reason='The real part of expm1(z) bad at these points')
    def test_expm1_complex_hard(self):
        y = np.array([0.1, 0.2, 0.3, 5, 11, 20])
        x = -np.log(np.cos(y))
        z = x + 1j * y
        expected = np.array([-5.5507901846769623e-17 + 0.10033467208545054j, 2.4289354732893695e-18 + 0.20271003550867248j, 4.523550026258577e-17 + 0.3093362496096232j, 7.8234305217489e-17 - 3.3805150062465863j, -1.3685191953697676e-16 - 225.95084645419513j, 8.717562048129105e-17 + 2.237160944224742j])
        found = cephes.expm1(z)
        assert_array_almost_equal_nulp(found.imag, expected.imag, 3)
        assert_array_almost_equal_nulp(found.real, expected.real, 20)

    def test_fdtr(self):
        assert_equal(cephes.fdtr(1, 1, 0), 0.0)
        assert_allclose(cephes.fdtr(1e-06, 5, 10), 0.9999940790193488, rtol=1e-12)

    def test_fdtrc(self):
        assert_equal(cephes.fdtrc(1, 1, 0), 1.0)
        assert_allclose(cephes.fdtrc(2, 0.1, 10000000000.0), 0.2722378462129351, rtol=1e-12)

    def test_fdtri(self):
        assert_allclose(cephes.fdtri(1, 1, [0.499, 0.501]), array([0.9937365, 1.00630298]), rtol=1e-06)
        p = 0.8756751669632106
        assert_allclose(cephes.fdtri(0.1, 1, p), 3, rtol=1e-12)

    @pytest.mark.xfail(reason='Returns nan on i686.')
    def test_fdtri_mysterious_failure(self):
        assert_allclose(cephes.fdtri(1, 1, 0.5), 1)

    def test_fdtridfd(self):
        assert_equal(cephes.fdtridfd(1, 0, 0), 5.0)

    def test_fresnel(self):
        assert_equal(cephes.fresnel(0), (0.0, 0.0))

    def test_gamma(self):
        assert_equal(cephes.gamma(5), 24.0)

    def test_gammainccinv(self):
        assert_equal(cephes.gammainccinv(5, 1), 0.0)

    def test_gammaln(self):
        cephes.gammaln(10)

    def test_gammasgn(self):
        vals = np.array([-4, -3.5, -2.3, 1, 4.2], np.float64)
        assert_array_equal(cephes.gammasgn(vals), np.sign(cephes.rgamma(vals)))

    def test_gdtr(self):
        assert_equal(cephes.gdtr(1, 1, 0), 0.0)

    def test_gdtr_inf(self):
        assert_equal(cephes.gdtr(1, 1, np.inf), 1.0)

    def test_gdtrc(self):
        assert_equal(cephes.gdtrc(1, 1, 0), 1.0)

    def test_gdtria(self):
        assert_equal(cephes.gdtria(0, 1, 1), 0.0)

    def test_gdtrib(self):
        cephes.gdtrib(1, 0, 1)

    def test_gdtrix(self):
        cephes.gdtrix(1, 1, 0.1)

    def test_hankel1(self):
        cephes.hankel1(1, 1)

    def test_hankel1e(self):
        cephes.hankel1e(1, 1)

    def test_hankel2(self):
        cephes.hankel2(1, 1)

    def test_hankel2e(self):
        cephes.hankel2e(1, 1)

    def test_hyp1f1(self):
        assert_approx_equal(cephes.hyp1f1(1, 1, 1), exp(1.0))
        assert_approx_equal(cephes.hyp1f1(3, 4, -6), 0.026056422099537253)
        cephes.hyp1f1(1, 1, 1)

    def test_hyp2f1(self):
        assert_equal(cephes.hyp2f1(1, 1, 1, 0), 1.0)

    def test_i0(self):
        assert_equal(cephes.i0(0), 1.0)

    def test_i0e(self):
        assert_equal(cephes.i0e(0), 1.0)

    def test_i1(self):
        assert_equal(cephes.i1(0), 0.0)

    def test_i1e(self):
        assert_equal(cephes.i1e(0), 0.0)

    def test_it2i0k0(self):
        cephes.it2i0k0(1)

    def test_it2j0y0(self):
        cephes.it2j0y0(1)

    def test_it2struve0(self):
        cephes.it2struve0(1)

    def test_itairy(self):
        cephes.itairy(1)

    def test_iti0k0(self):
        assert_equal(cephes.iti0k0(0), (0.0, 0.0))

    def test_itj0y0(self):
        assert_equal(cephes.itj0y0(0), (0.0, 0.0))

    def test_itmodstruve0(self):
        assert_equal(cephes.itmodstruve0(0), 0.0)

    def test_itstruve0(self):
        assert_equal(cephes.itstruve0(0), 0.0)

    def test_iv(self):
        assert_equal(cephes.iv(1, 0), 0.0)

    def test_ive(self):
        assert_equal(cephes.ive(1, 0), 0.0)

    def test_j0(self):
        assert_equal(cephes.j0(0), 1.0)

    def test_j1(self):
        assert_equal(cephes.j1(0), 0.0)

    def test_jn(self):
        assert_equal(cephes.jn(0, 0), 1.0)

    def test_jv(self):
        assert_equal(cephes.jv(0, 0), 1.0)

    def test_jve(self):
        assert_equal(cephes.jve(0, 0), 1.0)

    def test_k0(self):
        cephes.k0(2)

    def test_k0e(self):
        cephes.k0e(2)

    def test_k1(self):
        cephes.k1(2)

    def test_k1e(self):
        cephes.k1e(2)

    def test_kei(self):
        cephes.kei(2)

    def test_keip(self):
        assert_equal(cephes.keip(0), 0.0)

    def test_ker(self):
        cephes.ker(2)

    def test_kerp(self):
        cephes.kerp(2)

    def test_kelvin(self):
        cephes.kelvin(2)

    def test_kn(self):
        cephes.kn(1, 1)

    def test_kolmogi(self):
        assert_equal(cephes.kolmogi(1), 0.0)
        assert_(np.isnan(cephes.kolmogi(np.nan)))

    def test_kolmogorov(self):
        assert_equal(cephes.kolmogorov(0), 1.0)

    def test_kolmogp(self):
        assert_equal(cephes._kolmogp(0), -0.0)

    def test_kolmogc(self):
        assert_equal(cephes._kolmogc(0), 0.0)

    def test_kolmogci(self):
        assert_equal(cephes._kolmogci(0), 0.0)
        assert_(np.isnan(cephes._kolmogci(np.nan)))

    def test_kv(self):
        cephes.kv(1, 1)

    def test_kve(self):
        cephes.kve(1, 1)

    def test_log1p(self):
        log1p = cephes.log1p
        assert_equal(log1p(0), 0.0)
        assert_equal(log1p(-1), -np.inf)
        assert_equal(log1p(-2), np.nan)
        assert_equal(log1p(np.inf), np.inf)

    def test_log1p_complex(self):
        log1p = cephes.log1p
        c = complex
        assert_equal(log1p(0 + 0j), 0 + 0j)
        assert_equal(log1p(c(-1, 0)), c(-np.inf, 0))
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, 'invalid value encountered in multiply')
            assert_allclose(log1p(c(1, np.inf)), c(np.inf, np.pi / 2))
            assert_equal(log1p(c(1, np.nan)), c(np.nan, np.nan))
            assert_allclose(log1p(c(-np.inf, 1)), c(np.inf, np.pi))
            assert_equal(log1p(c(np.inf, 1)), c(np.inf, 0))
            assert_allclose(log1p(c(-np.inf, np.inf)), c(np.inf, 3 * np.pi / 4))
            assert_allclose(log1p(c(np.inf, np.inf)), c(np.inf, np.pi / 4))
            assert_equal(log1p(c(np.inf, np.nan)), c(np.inf, np.nan))
            assert_equal(log1p(c(-np.inf, np.nan)), c(np.inf, np.nan))
            assert_equal(log1p(c(np.nan, np.inf)), c(np.inf, np.nan))
            assert_equal(log1p(c(np.nan, 1)), c(np.nan, np.nan))
            assert_equal(log1p(c(np.nan, np.nan)), c(np.nan, np.nan))

    def test_lpmv(self):
        assert_equal(cephes.lpmv(0, 0, 1), 1.0)

    def test_mathieu_a(self):
        assert_equal(cephes.mathieu_a(1, 0), 1.0)

    def test_mathieu_b(self):
        assert_equal(cephes.mathieu_b(1, 0), 1.0)

    def test_mathieu_cem(self):
        assert_equal(cephes.mathieu_cem(1, 0, 0), (1.0, 0.0))

        @np.vectorize
        def ce_smallq(m, q, z):
            z *= np.pi / 180
            if m == 0:
                return 2 ** (-0.5) * (1 - 0.5 * q * cos(2 * z))
            elif m == 1:
                return cos(z) - q / 8 * cos(3 * z)
            elif m == 2:
                return cos(2 * z) - q * (cos(4 * z) / 12 - 1 / 4)
            else:
                return cos(m * z) - q * (cos((m + 2) * z) / (4 * (m + 1)) - cos((m - 2) * z) / (4 * (m - 1)))
        m = np.arange(0, 100)
        q = np.r_[0, np.logspace(-30, -9, 10)]
        assert_allclose(cephes.mathieu_cem(m[:, None], q[None, :], 0.123)[0], ce_smallq(m[:, None], q[None, :], 0.123), rtol=1e-14, atol=0)

    def test_mathieu_sem(self):
        assert_equal(cephes.mathieu_sem(1, 0, 0), (0.0, 1.0))

        @np.vectorize
        def se_smallq(m, q, z):
            z *= np.pi / 180
            if m == 1:
                return sin(z) - q / 8 * sin(3 * z)
            elif m == 2:
                return sin(2 * z) - q * sin(4 * z) / 12
            else:
                return sin(m * z) - q * (sin((m + 2) * z) / (4 * (m + 1)) - sin((m - 2) * z) / (4 * (m - 1)))
        m = np.arange(1, 100)
        q = np.r_[0, np.logspace(-30, -9, 10)]
        assert_allclose(cephes.mathieu_sem(m[:, None], q[None, :], 0.123)[0], se_smallq(m[:, None], q[None, :], 0.123), rtol=1e-14, atol=0)

    def test_mathieu_modcem1(self):
        assert_equal(cephes.mathieu_modcem1(1, 0, 0), (0.0, 0.0))

    def test_mathieu_modcem2(self):
        cephes.mathieu_modcem2(1, 1, 1)
        m = np.arange(0, 4)[:, None, None]
        q = np.r_[np.logspace(-2, 2, 10)][None, :, None]
        z = np.linspace(0, 1, 7)[None, None, :]
        y1 = cephes.mathieu_modcem2(m, q, -z)[0]
        fr = -cephes.mathieu_modcem2(m, q, 0)[0] / cephes.mathieu_modcem1(m, q, 0)[0]
        y2 = -cephes.mathieu_modcem2(m, q, z)[0] - 2 * fr * cephes.mathieu_modcem1(m, q, z)[0]
        assert_allclose(y1, y2, rtol=1e-10)

    def test_mathieu_modsem1(self):
        assert_equal(cephes.mathieu_modsem1(1, 0, 0), (0.0, 0.0))

    def test_mathieu_modsem2(self):
        cephes.mathieu_modsem2(1, 1, 1)
        m = np.arange(1, 4)[:, None, None]
        q = np.r_[np.logspace(-2, 2, 10)][None, :, None]
        z = np.linspace(0, 1, 7)[None, None, :]
        y1 = cephes.mathieu_modsem2(m, q, -z)[0]
        fr = cephes.mathieu_modsem2(m, q, 0)[1] / cephes.mathieu_modsem1(m, q, 0)[1]
        y2 = cephes.mathieu_modsem2(m, q, z)[0] - 2 * fr * cephes.mathieu_modsem1(m, q, z)[0]
        assert_allclose(y1, y2, rtol=1e-10)

    def test_mathieu_overflow(self):
        assert_equal(cephes.mathieu_cem(10000, 0, 1.3), (np.nan, np.nan))
        assert_equal(cephes.mathieu_sem(10000, 0, 1.3), (np.nan, np.nan))
        assert_equal(cephes.mathieu_cem(10000, 1.5, 1.3), (np.nan, np.nan))
        assert_equal(cephes.mathieu_sem(10000, 1.5, 1.3), (np.nan, np.nan))
        assert_equal(cephes.mathieu_modcem1(10000, 1.5, 1.3), (np.nan, np.nan))
        assert_equal(cephes.mathieu_modsem1(10000, 1.5, 1.3), (np.nan, np.nan))
        assert_equal(cephes.mathieu_modcem2(10000, 1.5, 1.3), (np.nan, np.nan))
        assert_equal(cephes.mathieu_modsem2(10000, 1.5, 1.3), (np.nan, np.nan))

    def test_mathieu_ticket_1847(self):
        for k in range(60):
            v = cephes.mathieu_modsem2(2, 100, -1)
            assert_allclose(v[0], 0.1431742913063671, rtol=1e-10)
            assert_allclose(v[1], 0.9017807375832909, rtol=0.0001)

    def test_modfresnelm(self):
        cephes.modfresnelm(0)

    def test_modfresnelp(self):
        cephes.modfresnelp(0)

    def test_modstruve(self):
        assert_equal(cephes.modstruve(1, 0), 0.0)

    def test_nbdtr(self):
        assert_equal(cephes.nbdtr(1, 1, 1), 1.0)

    def test_nbdtrc(self):
        assert_equal(cephes.nbdtrc(1, 1, 1), 0.0)

    def test_nbdtri(self):
        assert_equal(cephes.nbdtri(1, 1, 1), 1.0)

    def test_nbdtrik(self):
        cephes.nbdtrik(1, 0.4, 0.5)

    def test_nbdtrin(self):
        assert_equal(cephes.nbdtrin(1, 0, 0), 5.0)

    def test_ncfdtr(self):
        assert_equal(cephes.ncfdtr(1, 1, 1, 0), 0.0)

    def test_ncfdtri(self):
        assert_equal(cephes.ncfdtri(1, 1, 1, 0), 0.0)
        f = [0.5, 1, 1.5]
        p = cephes.ncfdtr(2, 3, 1.5, f)
        assert_allclose(cephes.ncfdtri(2, 3, 1.5, p), f)

    def test_ncfdtridfd(self):
        dfd = [1, 2, 3]
        p = cephes.ncfdtr(2, dfd, 0.25, 15)
        assert_allclose(cephes.ncfdtridfd(2, p, 0.25, 15), dfd)

    def test_ncfdtridfn(self):
        dfn = [0.1, 1, 2, 3, 10000.0]
        p = cephes.ncfdtr(dfn, 2, 0.25, 15)
        assert_allclose(cephes.ncfdtridfn(p, 2, 0.25, 15), dfn, rtol=1e-05)

    def test_ncfdtrinc(self):
        nc = [0.5, 1.5, 2.0]
        p = cephes.ncfdtr(2, 3, nc, 15)
        assert_allclose(cephes.ncfdtrinc(2, 3, p, 15), nc)

    def test_nctdtr(self):
        assert_equal(cephes.nctdtr(1, 0, 0), 0.5)
        assert_equal(cephes.nctdtr(9, 65536, 45), 0.0)
        assert_approx_equal(cephes.nctdtr(np.inf, 1.0, 1.0), 0.5, 5)
        assert_(np.isnan(cephes.nctdtr(2.0, np.inf, 10.0)))
        assert_approx_equal(cephes.nctdtr(2.0, 1.0, np.inf), 1.0)
        assert_(np.isnan(cephes.nctdtr(np.nan, 1.0, 1.0)))
        assert_(np.isnan(cephes.nctdtr(2.0, np.nan, 1.0)))
        assert_(np.isnan(cephes.nctdtr(2.0, 1.0, np.nan)))

    def test_nctdtridf(self):
        cephes.nctdtridf(1, 0.5, 0)

    def test_nctdtrinc(self):
        cephes.nctdtrinc(1, 0, 0)

    def test_nctdtrit(self):
        cephes.nctdtrit(0.1, 0.2, 0.5)

    def test_nrdtrimn(self):
        assert_approx_equal(cephes.nrdtrimn(0.5, 1, 1), 1.0)

    def test_nrdtrisd(self):
        assert_allclose(cephes.nrdtrisd(0.5, 0.5, 0.5), 0.0, atol=0, rtol=0)

    def test_obl_ang1(self):
        cephes.obl_ang1(1, 1, 1, 0)

    def test_obl_ang1_cv(self):
        result = cephes.obl_ang1_cv(1, 1, 1, 1, 0)
        assert_almost_equal(result[0], 1.0)
        assert_almost_equal(result[1], 0.0)

    def test_obl_cv(self):
        assert_equal(cephes.obl_cv(1, 1, 0), 2.0)

    def test_obl_rad1(self):
        cephes.obl_rad1(1, 1, 1, 0)

    def test_obl_rad1_cv(self):
        cephes.obl_rad1_cv(1, 1, 1, 1, 0)

    def test_obl_rad2(self):
        cephes.obl_rad2(1, 1, 1, 0)

    def test_obl_rad2_cv(self):
        cephes.obl_rad2_cv(1, 1, 1, 1, 0)

    def test_pbdv(self):
        assert_equal(cephes.pbdv(1, 0), (0.0, 1.0))

    def test_pbvv(self):
        cephes.pbvv(1, 0)

    def test_pbwa(self):
        cephes.pbwa(1, 0)

    def test_pdtr(self):
        val = cephes.pdtr(0, 1)
        assert_almost_equal(val, np.exp(-1))
        val = cephes.pdtr([0, 1, 2], 0)
        assert_array_equal(val, [1, 1, 1])

    def test_pdtrc(self):
        val = cephes.pdtrc(0, 1)
        assert_almost_equal(val, 1 - np.exp(-1))
        val = cephes.pdtrc([0, 1, 2], 0.0)
        assert_array_equal(val, [0, 0, 0])

    def test_pdtri(self):
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, 'floating point number truncated to an integer')
            cephes.pdtri(0.5, 0.5)

    def test_pdtrik(self):
        k = cephes.pdtrik(0.5, 1)
        assert_almost_equal(cephes.gammaincc(k + 1, 1), 0.5)
        k = cephes.pdtrik([[0], [0.25], [0.95]], [0, 1e-20, 1e-06])
        assert_array_equal(k, np.zeros((3, 3)))

    def test_pro_ang1(self):
        cephes.pro_ang1(1, 1, 1, 0)

    def test_pro_ang1_cv(self):
        assert_array_almost_equal(cephes.pro_ang1_cv(1, 1, 1, 1, 0), array((1.0, 0.0)))

    def test_pro_cv(self):
        assert_equal(cephes.pro_cv(1, 1, 0), 2.0)

    def test_pro_rad1(self):
        cephes.pro_rad1(1, 1, 1, 0.1)

    def test_pro_rad1_cv(self):
        cephes.pro_rad1_cv(1, 1, 1, 1, 0)

    def test_pro_rad2(self):
        cephes.pro_rad2(1, 1, 1, 0)

    def test_pro_rad2_cv(self):
        cephes.pro_rad2_cv(1, 1, 1, 1, 0)

    def test_psi(self):
        cephes.psi(1)

    def test_radian(self):
        assert_equal(cephes.radian(0, 0, 0), 0)

    def test_rgamma(self):
        assert_equal(cephes.rgamma(1), 1.0)

    def test_round(self):
        assert_equal(cephes.round(3.4), 3.0)
        assert_equal(cephes.round(-3.4), -3.0)
        assert_equal(cephes.round(3.6), 4.0)
        assert_equal(cephes.round(-3.6), -4.0)
        assert_equal(cephes.round(3.5), 4.0)
        assert_equal(cephes.round(-3.5), -4.0)

    def test_shichi(self):
        cephes.shichi(1)

    def test_sici(self):
        cephes.sici(1)
        s, c = cephes.sici(np.inf)
        assert_almost_equal(s, np.pi * 0.5)
        assert_almost_equal(c, 0)
        s, c = cephes.sici(-np.inf)
        assert_almost_equal(s, -np.pi * 0.5)
        assert_(np.isnan(c), 'cosine integral(-inf) is not nan')

    def test_sindg(self):
        assert_equal(cephes.sindg(90), 1.0)

    def test_smirnov(self):
        assert_equal(cephes.smirnov(1, 0.1), 0.9)
        assert_(np.isnan(cephes.smirnov(1, np.nan)))

    def test_smirnovp(self):
        assert_equal(cephes._smirnovp(1, 0.1), -1)
        assert_equal(cephes._smirnovp(2, 0.75), -2 * 0.25 ** (2 - 1))
        assert_equal(cephes._smirnovp(3, 0.75), -3 * 0.25 ** (3 - 1))
        assert_(np.isnan(cephes._smirnovp(1, np.nan)))

    def test_smirnovc(self):
        assert_equal(cephes._smirnovc(1, 0.1), 0.1)
        assert_(np.isnan(cephes._smirnovc(1, np.nan)))
        x10 = np.linspace(0, 1, 11, endpoint=True)
        assert_almost_equal(cephes._smirnovc(3, x10), 1 - cephes.smirnov(3, x10))
        x4 = np.linspace(0, 1, 5, endpoint=True)
        assert_almost_equal(cephes._smirnovc(4, x4), 1 - cephes.smirnov(4, x4))

    def test_smirnovi(self):
        assert_almost_equal(cephes.smirnov(1, cephes.smirnovi(1, 0.4)), 0.4)
        assert_almost_equal(cephes.smirnov(1, cephes.smirnovi(1, 0.6)), 0.6)
        assert_(np.isnan(cephes.smirnovi(1, np.nan)))

    def test_smirnovci(self):
        assert_almost_equal(cephes._smirnovc(1, cephes._smirnovci(1, 0.4)), 0.4)
        assert_almost_equal(cephes._smirnovc(1, cephes._smirnovci(1, 0.6)), 0.6)
        assert_(np.isnan(cephes._smirnovci(1, np.nan)))

    def test_spence(self):
        assert_equal(cephes.spence(1), 0.0)

    def test_stdtr(self):
        assert_equal(cephes.stdtr(1, 0), 0.5)
        assert_almost_equal(cephes.stdtr(1, 1), 0.75)
        assert_almost_equal(cephes.stdtr(1, 2), 0.852416382349)

    def test_stdtridf(self):
        cephes.stdtridf(0.7, 1)

    def test_stdtrit(self):
        cephes.stdtrit(1, 0.7)

    def test_struve(self):
        assert_equal(cephes.struve(0, 0), 0.0)

    def test_tandg(self):
        assert_equal(cephes.tandg(45), 1.0)

    def test_tklmbda(self):
        assert_almost_equal(cephes.tklmbda(1, 1), 1.0)

    def test_y0(self):
        cephes.y0(1)

    def test_y1(self):
        cephes.y1(1)

    def test_yn(self):
        cephes.yn(1, 1)

    def test_yv(self):
        cephes.yv(1, 1)

    def test_yve(self):
        cephes.yve(1, 1)

    def test_wofz(self):
        z = [complex(624.2, -0.26123), complex(-0.4, 3.0), complex(0.6, 2.0), complex(-1.0, 1.0), complex(-1.0, -9.0), complex(-1.0, 9.0), complex(-2.34545e-08, 1.1234), complex(-3.0, 5.1), complex(-53, 30.1), complex(0.0, 0.12345), complex(11, 1), complex(-22, -2), complex(9, -28), complex(21, -33), complex(100000.0, 100000.0), complex(100000000000000.0, 100000000000000.0)]
        w = [complex(-3.7827024551898053e-07, 0.0009038612764331721), complex(0.1764906227004817, -0.021465505394684576), complex(0.2410250715772692, 0.060875796634280895), complex(0.3047442052569126, -0.20821893820283163), complex(7.317131068972378e+34, 8.321873499714403e+34), complex(0.06156985072363237, -0.00676005783716575), complex(0.3960793007699875, -5.593152259116645e-09), complex(0.08217199226739448, -0.0470129108764361), complex(0.0045724600035028165, -0.008049007914116918), complex(0.8746342859608053, 0.0), complex(0.004681901649654442, 0.05107355639013062), complex(-0.002319317520018762, -0.025460054739731557), complex(9.114633684056372e+304, 3.971018071452634e+305), complex(-4.49272078577156e+281, -2.8019591213423078e+281), complex(2.8209479178093053e-06, 2.8209479176682578e-06), complex(2.8209479177387813e-15, 2.8209479177387813e-15)]
        assert_func_equal(cephes.wofz, w, z, rtol=1e-13)