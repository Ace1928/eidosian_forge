import hashlib
import pickle
import sys
import warnings
import numpy as np
import pytest
from numpy.testing import (
from numpy.random import MT19937, PCG64
from numpy import random
class TestBroadcast:

    def setup_method(self):
        self.seed = 123456789

    def set_seed(self):
        random.seed(self.seed)

    def test_uniform(self):
        low = [0]
        high = [1]
        uniform = random.uniform
        desired = np.array([0.532833024789759, 0.5341366008904166, 0.509553035526467])
        self.set_seed()
        actual = uniform(low * 3, high)
        assert_array_almost_equal(actual, desired, decimal=14)
        self.set_seed()
        actual = uniform(low, high * 3)
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_normal(self):
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        normal = random.normal
        desired = np.array([2.2129019979039612, 2.128397797652002, 1.8417114045748335])
        self.set_seed()
        actual = normal(loc * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, normal, loc * 3, bad_scale)
        self.set_seed()
        actual = normal(loc, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, normal, loc, bad_scale * 3)

    def test_beta(self):
        a = [1]
        b = [2]
        bad_a = [-1]
        bad_b = [-2]
        beta = random.beta
        desired = np.array([0.19843558305989056, 0.07523033640942364, 0.24976865978980844])
        self.set_seed()
        actual = beta(a * 3, b)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, beta, bad_a * 3, b)
        assert_raises(ValueError, beta, a * 3, bad_b)
        self.set_seed()
        actual = beta(a, b * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, beta, bad_a, b * 3)
        assert_raises(ValueError, beta, a, bad_b * 3)

    def test_exponential(self):
        scale = [1]
        bad_scale = [-1]
        exponential = random.exponential
        desired = np.array([0.7610685365884524, 0.7638628227869165, 0.712438131258918])
        self.set_seed()
        actual = exponential(scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, exponential, bad_scale * 3)

    def test_standard_gamma(self):
        shape = [1]
        bad_shape = [-1]
        std_gamma = random.standard_gamma
        desired = np.array([0.7610685365884524, 0.7638628227869165, 0.712438131258918])
        self.set_seed()
        actual = std_gamma(shape * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, std_gamma, bad_shape * 3)

    def test_gamma(self):
        shape = [1]
        scale = [2]
        bad_shape = [-1]
        bad_scale = [-2]
        gamma = random.gamma
        desired = np.array([1.5221370731769048, 1.527725645573833, 1.424876262517836])
        self.set_seed()
        actual = gamma(shape * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, gamma, bad_shape * 3, scale)
        assert_raises(ValueError, gamma, shape * 3, bad_scale)
        self.set_seed()
        actual = gamma(shape, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, gamma, bad_shape, scale * 3)
        assert_raises(ValueError, gamma, shape, bad_scale * 3)

    def test_f(self):
        dfnum = [1]
        dfden = [2]
        bad_dfnum = [-1]
        bad_dfden = [-2]
        f = random.f
        desired = np.array([0.800389516382648, 0.8676871963536351, 2.72510951683868])
        self.set_seed()
        actual = f(dfnum * 3, dfden)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, f, bad_dfnum * 3, dfden)
        assert_raises(ValueError, f, dfnum * 3, bad_dfden)
        self.set_seed()
        actual = f(dfnum, dfden * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, f, bad_dfnum, dfden * 3)
        assert_raises(ValueError, f, dfnum, bad_dfden * 3)

    def test_noncentral_f(self):
        dfnum = [2]
        dfden = [3]
        nonc = [4]
        bad_dfnum = [0]
        bad_dfden = [-1]
        bad_nonc = [-2]
        nonc_f = random.noncentral_f
        desired = np.array([9.139394326370521, 13.025456344595602, 8.801809835910055])
        self.set_seed()
        actual = nonc_f(dfnum * 3, dfden, nonc)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert np.all(np.isnan(nonc_f(dfnum, dfden, [np.nan] * 3)))
        assert_raises(ValueError, nonc_f, bad_dfnum * 3, dfden, nonc)
        assert_raises(ValueError, nonc_f, dfnum * 3, bad_dfden, nonc)
        assert_raises(ValueError, nonc_f, dfnum * 3, dfden, bad_nonc)
        self.set_seed()
        actual = nonc_f(dfnum, dfden * 3, nonc)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, nonc_f, bad_dfnum, dfden * 3, nonc)
        assert_raises(ValueError, nonc_f, dfnum, bad_dfden * 3, nonc)
        assert_raises(ValueError, nonc_f, dfnum, dfden * 3, bad_nonc)
        self.set_seed()
        actual = nonc_f(dfnum, dfden, nonc * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, nonc_f, bad_dfnum, dfden, nonc * 3)
        assert_raises(ValueError, nonc_f, dfnum, bad_dfden, nonc * 3)
        assert_raises(ValueError, nonc_f, dfnum, dfden, bad_nonc * 3)

    def test_noncentral_f_small_df(self):
        self.set_seed()
        desired = np.array([6.869638627492048, 0.785880199263955])
        actual = random.noncentral_f(0.9, 0.9, 2, size=2)
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_chisquare(self):
        df = [1]
        bad_df = [-1]
        chisquare = random.chisquare
        desired = np.array([0.5702280113308829, 0.5194770210884078, 0.1320969254923558])
        self.set_seed()
        actual = chisquare(df * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, chisquare, bad_df * 3)

    def test_noncentral_chisquare(self):
        df = [1]
        nonc = [2]
        bad_df = [-1]
        bad_nonc = [-2]
        nonc_chi = random.noncentral_chisquare
        desired = np.array([9.001559946791376, 4.580413504971874, 6.087230243283456])
        self.set_seed()
        actual = nonc_chi(df * 3, nonc)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, nonc_chi, bad_df * 3, nonc)
        assert_raises(ValueError, nonc_chi, df * 3, bad_nonc)
        self.set_seed()
        actual = nonc_chi(df, nonc * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, nonc_chi, bad_df, nonc * 3)
        assert_raises(ValueError, nonc_chi, df, bad_nonc * 3)

    def test_standard_t(self):
        df = [1]
        bad_df = [-1]
        t = random.standard_t
        desired = np.array([3.0702872575217643, 5.856072516736161, 1.0274791436474273])
        self.set_seed()
        actual = t(df * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, t, bad_df * 3)
        assert_raises(ValueError, random.standard_t, bad_df * 3)

    def test_vonmises(self):
        mu = [2]
        kappa = [1]
        bad_kappa = [-1]
        vonmises = random.vonmises
        desired = np.array([2.9883443664201312, -2.7064099483995943, -1.8672476700665914])
        self.set_seed()
        actual = vonmises(mu * 3, kappa)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, vonmises, mu * 3, bad_kappa)
        self.set_seed()
        actual = vonmises(mu, kappa * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, vonmises, mu, bad_kappa * 3)

    def test_pareto(self):
        a = [1]
        bad_a = [-1]
        pareto = random.pareto
        desired = np.array([1.1405622680198362, 1.146551976204453, 1.0389564467453547])
        self.set_seed()
        actual = pareto(a * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, pareto, bad_a * 3)
        assert_raises(ValueError, random.pareto, bad_a * 3)

    def test_weibull(self):
        a = [1]
        bad_a = [-1]
        weibull = random.weibull
        desired = np.array([0.7610685365884524, 0.7638628227869165, 0.712438131258918])
        self.set_seed()
        actual = weibull(a * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, weibull, bad_a * 3)
        assert_raises(ValueError, random.weibull, bad_a * 3)

    def test_power(self):
        a = [1]
        bad_a = [-1]
        power = random.power
        desired = np.array([0.532833024789759, 0.5341366008904166, 0.509553035526467])
        self.set_seed()
        actual = power(a * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, power, bad_a * 3)
        assert_raises(ValueError, random.power, bad_a * 3)

    def test_laplace(self):
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        laplace = random.laplace
        desired = np.array([0.06792135602850716, 0.07071564222697133, 0.019290950698972624])
        self.set_seed()
        actual = laplace(loc * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, laplace, loc * 3, bad_scale)
        self.set_seed()
        actual = laplace(loc, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, laplace, loc, bad_scale * 3)

    def test_gumbel(self):
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        gumbel = random.gumbel
        desired = np.array([0.2730318639556768, 0.26936705726291116, 0.3390622039303794])
        self.set_seed()
        actual = gumbel(loc * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, gumbel, loc * 3, bad_scale)
        self.set_seed()
        actual = gumbel(loc, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, gumbel, loc, bad_scale * 3)

    def test_logistic(self):
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        logistic = random.logistic
        desired = np.array([0.1315213583758617, 0.13675915696285773, 0.038216792802833396])
        self.set_seed()
        actual = logistic(loc * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, logistic, loc * 3, bad_scale)
        self.set_seed()
        actual = logistic(loc, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, logistic, loc, bad_scale * 3)
        assert_equal(random.logistic(1.0, 0.0), 1.0)

    def test_lognormal(self):
        mean = [0]
        sigma = [1]
        bad_sigma = [-1]
        lognormal = random.lognormal
        desired = np.array([9.142208604484843, 8.401395287012626, 6.307323411657867])
        self.set_seed()
        actual = lognormal(mean * 3, sigma)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, lognormal, mean * 3, bad_sigma)
        assert_raises(ValueError, random.lognormal, mean * 3, bad_sigma)
        self.set_seed()
        actual = lognormal(mean, sigma * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, lognormal, mean, bad_sigma * 3)
        assert_raises(ValueError, random.lognormal, mean, bad_sigma * 3)

    def test_rayleigh(self):
        scale = [1]
        bad_scale = [-1]
        rayleigh = random.rayleigh
        desired = np.array([1.2337491937897689, 1.2360119924878694, 1.193681809578179])
        self.set_seed()
        actual = rayleigh(scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, rayleigh, bad_scale * 3)

    def test_wald(self):
        mean = [0.5]
        scale = [1]
        bad_mean = [0]
        bad_scale = [-2]
        wald = random.wald
        desired = np.array([0.11873681120271318, 0.12450084820795027, 0.9096122728408238])
        self.set_seed()
        actual = wald(mean * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, wald, bad_mean * 3, scale)
        assert_raises(ValueError, wald, mean * 3, bad_scale)
        assert_raises(ValueError, random.wald, bad_mean * 3, scale)
        assert_raises(ValueError, random.wald, mean * 3, bad_scale)
        self.set_seed()
        actual = wald(mean, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, wald, bad_mean, scale * 3)
        assert_raises(ValueError, wald, mean, bad_scale * 3)
        assert_raises(ValueError, wald, 0.0, 1)
        assert_raises(ValueError, wald, 0.5, 0.0)

    def test_triangular(self):
        left = [1]
        right = [3]
        mode = [2]
        bad_left_one = [3]
        bad_mode_one = [4]
        bad_left_two, bad_mode_two = right * 2
        triangular = random.triangular
        desired = np.array([2.03339048710429, 2.0347400359389356, 2.0095991069536208])
        self.set_seed()
        actual = triangular(left * 3, mode, right)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, triangular, bad_left_one * 3, mode, right)
        assert_raises(ValueError, triangular, left * 3, bad_mode_one, right)
        assert_raises(ValueError, triangular, bad_left_two * 3, bad_mode_two, right)
        self.set_seed()
        actual = triangular(left, mode * 3, right)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, triangular, bad_left_one, mode * 3, right)
        assert_raises(ValueError, triangular, left, bad_mode_one * 3, right)
        assert_raises(ValueError, triangular, bad_left_two, bad_mode_two * 3, right)
        self.set_seed()
        actual = triangular(left, mode, right * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, triangular, bad_left_one, mode, right * 3)
        assert_raises(ValueError, triangular, left, bad_mode_one, right * 3)
        assert_raises(ValueError, triangular, bad_left_two, bad_mode_two, right * 3)
        assert_raises(ValueError, triangular, 10.0, 0.0, 20.0)
        assert_raises(ValueError, triangular, 10.0, 25.0, 20.0)
        assert_raises(ValueError, triangular, 10.0, 10.0, 10.0)

    def test_binomial(self):
        n = [1]
        p = [0.5]
        bad_n = [-1]
        bad_p_one = [-1]
        bad_p_two = [1.5]
        binom = random.binomial
        desired = np.array([1, 1, 1])
        self.set_seed()
        actual = binom(n * 3, p)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, binom, bad_n * 3, p)
        assert_raises(ValueError, binom, n * 3, bad_p_one)
        assert_raises(ValueError, binom, n * 3, bad_p_two)
        self.set_seed()
        actual = binom(n, p * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, binom, bad_n, p * 3)
        assert_raises(ValueError, binom, n, bad_p_one * 3)
        assert_raises(ValueError, binom, n, bad_p_two * 3)

    def test_negative_binomial(self):
        n = [1]
        p = [0.5]
        bad_n = [-1]
        bad_p_one = [-1]
        bad_p_two = [1.5]
        neg_binom = random.negative_binomial
        desired = np.array([1, 0, 1])
        self.set_seed()
        actual = neg_binom(n * 3, p)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, neg_binom, bad_n * 3, p)
        assert_raises(ValueError, neg_binom, n * 3, bad_p_one)
        assert_raises(ValueError, neg_binom, n * 3, bad_p_two)
        self.set_seed()
        actual = neg_binom(n, p * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, neg_binom, bad_n, p * 3)
        assert_raises(ValueError, neg_binom, n, bad_p_one * 3)
        assert_raises(ValueError, neg_binom, n, bad_p_two * 3)

    def test_poisson(self):
        max_lam = random.RandomState()._poisson_lam_max
        lam = [1]
        bad_lam_one = [-1]
        bad_lam_two = [max_lam * 2]
        poisson = random.poisson
        desired = np.array([1, 1, 0])
        self.set_seed()
        actual = poisson(lam * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, poisson, bad_lam_one * 3)
        assert_raises(ValueError, poisson, bad_lam_two * 3)

    def test_zipf(self):
        a = [2]
        bad_a = [0]
        zipf = random.zipf
        desired = np.array([2, 2, 1])
        self.set_seed()
        actual = zipf(a * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, zipf, bad_a * 3)
        with np.errstate(invalid='ignore'):
            assert_raises(ValueError, zipf, np.nan)
            assert_raises(ValueError, zipf, [0, 0, np.nan])

    def test_geometric(self):
        p = [0.5]
        bad_p_one = [-1]
        bad_p_two = [1.5]
        geom = random.geometric
        desired = np.array([2, 2, 2])
        self.set_seed()
        actual = geom(p * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, geom, bad_p_one * 3)
        assert_raises(ValueError, geom, bad_p_two * 3)

    def test_hypergeometric(self):
        ngood = [1]
        nbad = [2]
        nsample = [2]
        bad_ngood = [-1]
        bad_nbad = [-2]
        bad_nsample_one = [0]
        bad_nsample_two = [4]
        hypergeom = random.hypergeometric
        desired = np.array([1, 1, 1])
        self.set_seed()
        actual = hypergeom(ngood * 3, nbad, nsample)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, hypergeom, bad_ngood * 3, nbad, nsample)
        assert_raises(ValueError, hypergeom, ngood * 3, bad_nbad, nsample)
        assert_raises(ValueError, hypergeom, ngood * 3, nbad, bad_nsample_one)
        assert_raises(ValueError, hypergeom, ngood * 3, nbad, bad_nsample_two)
        self.set_seed()
        actual = hypergeom(ngood, nbad * 3, nsample)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, hypergeom, bad_ngood, nbad * 3, nsample)
        assert_raises(ValueError, hypergeom, ngood, bad_nbad * 3, nsample)
        assert_raises(ValueError, hypergeom, ngood, nbad * 3, bad_nsample_one)
        assert_raises(ValueError, hypergeom, ngood, nbad * 3, bad_nsample_two)
        self.set_seed()
        actual = hypergeom(ngood, nbad, nsample * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, hypergeom, bad_ngood, nbad, nsample * 3)
        assert_raises(ValueError, hypergeom, ngood, bad_nbad, nsample * 3)
        assert_raises(ValueError, hypergeom, ngood, nbad, bad_nsample_one * 3)
        assert_raises(ValueError, hypergeom, ngood, nbad, bad_nsample_two * 3)
        assert_raises(ValueError, hypergeom, -1, 10, 20)
        assert_raises(ValueError, hypergeom, 10, -1, 20)
        assert_raises(ValueError, hypergeom, 10, 10, 0)
        assert_raises(ValueError, hypergeom, 10, 10, 25)

    def test_logseries(self):
        p = [0.5]
        bad_p_one = [2]
        bad_p_two = [-1]
        logseries = random.logseries
        desired = np.array([1, 1, 1])
        self.set_seed()
        actual = logseries(p * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, logseries, bad_p_one * 3)
        assert_raises(ValueError, logseries, bad_p_two * 3)