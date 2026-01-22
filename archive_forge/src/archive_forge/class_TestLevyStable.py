import warnings
import re
import sys
import pickle
from pathlib import Path
import os
import json
import platform
from numpy.testing import (assert_equal, assert_array_equal,
import pytest
from pytest import raises as assert_raises
import numpy
import numpy as np
from numpy import typecodes, array
from numpy.lib.recfunctions import rec_append_fields
from scipy import special
from scipy._lib._util import check_random_state
from scipy.integrate import (IntegrationWarning, quad, trapezoid,
import scipy.stats as stats
from scipy.stats._distn_infrastructure import argsreduce
import scipy.stats.distributions
from scipy.special import xlogy, polygamma, entr
from scipy.stats._distr_params import distcont, invdistcont
from .test_discrete_basic import distdiscrete, invdistdiscrete
from scipy.stats._continuous_distns import FitDataError, _argus_phi
from scipy.optimize import root, fmin, differential_evolution
from itertools import product
class TestLevyStable:

    @pytest.fixture(autouse=True)
    def reset_levy_stable_params(self):
        """Setup default parameters for levy_stable generator"""
        stats.levy_stable.parameterization = 'S1'
        stats.levy_stable.cdf_default_method = 'piecewise'
        stats.levy_stable.pdf_default_method = 'piecewise'
        stats.levy_stable.quad_eps = stats._levy_stable._QUAD_EPS

    @pytest.fixture
    def nolan_pdf_sample_data(self):
        """Sample data points for pdf computed with Nolan's stablec

        See - http://fs2.american.edu/jpnolan/www/stable/stable.html

        There's a known limitation of Nolan's executable for alpha < 0.2.

        The data table loaded below is generated from Nolan's stablec
        with the following parameter space:

            alpha = 0.1, 0.2, ..., 2.0
            beta = -1.0, -0.9, ..., 1.0
            p = 0.01, 0.05, 0.1, 0.25, 0.35, 0.5,
        and the equivalent for the right tail

        Typically inputs for stablec:

            stablec.exe <<
            1 # pdf
            1 # Nolan S equivalent to S0 in scipy
            .25,2,.25 # alpha
            -1,-1,0 # beta
            -10,10,1 # x
            1,0 # gamma, delta
            2 # output file
        """
        data = np.load(Path(__file__).parent / 'data/levy_stable/stable-Z1-pdf-sample-data.npy')
        data = np.rec.fromarrays(data.T, names='x,p,alpha,beta,pct')
        return data

    @pytest.fixture
    def nolan_cdf_sample_data(self):
        """Sample data points for cdf computed with Nolan's stablec

        See - http://fs2.american.edu/jpnolan/www/stable/stable.html

        There's a known limitation of Nolan's executable for alpha < 0.2.

        The data table loaded below is generated from Nolan's stablec
        with the following parameter space:

            alpha = 0.1, 0.2, ..., 2.0
            beta = -1.0, -0.9, ..., 1.0
            p = 0.01, 0.05, 0.1, 0.25, 0.35, 0.5,

        and the equivalent for the right tail

        Ideally, Nolan's output for CDF values should match the percentile
        from where they have been sampled from. Even more so as we extract
        percentile x positions from stablec too. However, we note at places
        Nolan's stablec will produce absolute errors in order of 1e-5. We
        compare against his calculations here. In future, once we less
        reliant on Nolan's paper we might switch to comparing directly at
        percentiles (those x values being produced from some alternative
        means).

        Typically inputs for stablec:

            stablec.exe <<
            2 # cdf
            1 # Nolan S equivalent to S0 in scipy
            .25,2,.25 # alpha
            -1,-1,0 # beta
            -10,10,1 # x
            1,0 # gamma, delta
            2 # output file
        """
        data = np.load(Path(__file__).parent / 'data/levy_stable/stable-Z1-cdf-sample-data.npy')
        data = np.rec.fromarrays(data.T, names='x,p,alpha,beta,pct')
        return data

    @pytest.fixture
    def nolan_loc_scale_sample_data(self):
        """Sample data where loc, scale are different from 0, 1

        Data extracted in similar way to pdf/cdf above using
        Nolan's stablec but set to an arbitrary location scale of
        (2, 3) for various important parameters alpha, beta and for
        parameterisations S0 and S1.
        """
        data = np.load(Path(__file__).parent / 'data/levy_stable/stable-loc-scale-sample-data.npy')
        return data

    @pytest.mark.parametrize('sample_size', [pytest.param(50), pytest.param(1500, marks=pytest.mark.slow)])
    @pytest.mark.parametrize('parameterization', ['S0', 'S1'])
    @pytest.mark.parametrize('alpha,beta', [(1.0, 0), (1.0, -0.5), (1.5, 0), (1.9, 0.5)])
    @pytest.mark.parametrize('gamma,delta', [(1, 0), (3, 2)])
    def test_rvs(self, parameterization, alpha, beta, gamma, delta, sample_size):
        stats.levy_stable.parameterization = parameterization
        ls = stats.levy_stable(alpha=alpha, beta=beta, scale=gamma, loc=delta)
        _, p = stats.kstest(ls.rvs(size=sample_size, random_state=1234), ls.cdf)
        assert p > 0.05

    @pytest.mark.slow
    @pytest.mark.parametrize('beta', [0.5, 1])
    def test_rvs_alpha1(self, beta):
        """Additional test cases for rvs for alpha equal to 1."""
        np.random.seed(987654321)
        alpha = 1.0
        loc = 0.5
        scale = 1.5
        x = stats.levy_stable.rvs(alpha, beta, loc=loc, scale=scale, size=5000)
        stat, p = stats.kstest(x, 'levy_stable', args=(alpha, beta, loc, scale))
        assert p > 0.01

    def test_fit(self):
        x = [-0.05413, -0.05413, 0.0, 0.0, 0.0, 0.0, 0.00533, 0.00533, 0.00533, 0.00533, 0.00533, 0.03354, 0.03354, 0.03354, 0.03354, 0.03354, 0.05309, 0.05309, 0.05309, 0.05309, 0.05309]
        alpha1, beta1, loc1, scale1 = stats.levy_stable._fitstart(x)
        assert_allclose(alpha1, 1.48, rtol=0, atol=0.01)
        assert_almost_equal(beta1, -0.22, 2)
        assert_almost_equal(scale1, 0.01717, 4)
        assert_almost_equal(loc1, 0.00233, 2)
        x2 = x + [0.05309, 0.05309, 0.05309, 0.05309, 0.05309]
        alpha2, beta2, loc2, scale2 = stats.levy_stable._fitstart(x2)
        assert_equal(alpha2, 2)
        assert_equal(beta2, -1)
        assert_almost_equal(scale2, 0.02503, 4)
        assert_almost_equal(loc2, 0.03354, 4)

    @pytest.mark.xfail(reason='Unknown problem with fitstart.')
    @pytest.mark.parametrize('alpha,beta,delta,gamma', [(1.5, 0.4, 2, 3), (1.0, 0.4, 2, 3)])
    @pytest.mark.parametrize('parametrization', ['S0', 'S1'])
    def test_fit_rvs(self, alpha, beta, delta, gamma, parametrization):
        """Test that fit agrees with rvs for each parametrization."""
        stats.levy_stable.parametrization = parametrization
        data = stats.levy_stable.rvs(alpha, beta, loc=delta, scale=gamma, size=10000, random_state=1234)
        fit = stats.levy_stable._fitstart(data)
        alpha_obs, beta_obs, delta_obs, gamma_obs = fit
        assert_allclose([alpha, beta, delta, gamma], [alpha_obs, beta_obs, delta_obs, gamma_obs], rtol=0.01)

    def test_fit_beta_flip(self):
        x = np.array([1, 1, 3, 3, 10, 10, 10, 30, 30, 100, 100])
        alpha1, beta1, loc1, scale1 = stats.levy_stable._fitstart(x)
        alpha2, beta2, loc2, scale2 = stats.levy_stable._fitstart(-x)
        assert_equal(beta1, 1)
        assert loc1 != 0
        assert_almost_equal(alpha2, alpha1)
        assert_almost_equal(beta2, -beta1)
        assert_almost_equal(loc2, -loc1)
        assert_almost_equal(scale2, scale1)

    def test_fit_delta_shift(self):
        SHIFT = 1
        x = np.array([1, 1, 3, 3, 10, 10, 10, 30, 30, 100, 100])
        alpha1, beta1, loc1, scale1 = stats.levy_stable._fitstart(-x)
        alpha2, beta2, loc2, scale2 = stats.levy_stable._fitstart(-x + SHIFT)
        assert_almost_equal(alpha2, alpha1)
        assert_almost_equal(beta2, beta1)
        assert_almost_equal(loc2, loc1 + SHIFT)
        assert_almost_equal(scale2, scale1)

    def test_fit_loc_extrap(self):
        x = [1, 1, 3, 3, 10, 10, 10, 30, 30, 140, 140]
        alpha1, beta1, loc1, scale1 = stats.levy_stable._fitstart(x)
        assert alpha1 < 1, f'Expected alpha < 1, got {alpha1}'
        assert loc1 < min(x), f'Expected loc < {min(x)}, got {loc1}'
        x2 = [1, 1, 3, 3, 10, 10, 10, 30, 30, 130, 130]
        alpha2, beta2, loc2, scale2 = stats.levy_stable._fitstart(x2)
        assert alpha2 > 1, f'Expected alpha > 1, got {alpha2}'
        assert loc2 > max(x2), f'Expected loc > {max(x2)}, got {loc2}'

    @pytest.mark.parametrize('pct_range,alpha_range,beta_range', [pytest.param([0.01, 0.5, 0.99], [0.1, 1, 2], [-1, 0, 0.8]), pytest.param([0.01, 0.05, 0.5, 0.95, 0.99], [0.1, 0.5, 1, 1.5, 2], [-0.9, -0.5, 0, 0.3, 0.6, 1], marks=pytest.mark.slow), pytest.param([0.01, 0.05, 0.1, 0.25, 0.35, 0.5, 0.65, 0.75, 0.9, 0.95, 0.99], np.linspace(0.1, 2, 20), np.linspace(-1, 1, 21), marks=pytest.mark.xslow)])
    def test_pdf_nolan_samples(self, nolan_pdf_sample_data, pct_range, alpha_range, beta_range):
        """Test pdf values against Nolan's stablec.exe output"""
        data = nolan_pdf_sample_data
        uname = platform.uname()
        is_linux_32 = uname.system == 'Linux' and uname.machine == 'i686'
        platform_desc = '/'.join([uname.system, uname.machine, uname.processor])
        tests = [['dni', 1e-07, lambda r: np.isin(r['pct'], pct_range) & np.isin(r['alpha'], alpha_range) & np.isin(r['beta'], beta_range) & ~((r['beta'] == 0) & (r['pct'] == 0.5) | (r['beta'] >= 0.9) & (r['alpha'] >= 1.6) & (r['pct'] == 0.5) | (r['alpha'] <= 0.4) & np.isin(r['pct'], [0.01, 0.99]) | (r['alpha'] <= 0.3) & np.isin(r['pct'], [0.05, 0.95]) | (r['alpha'] <= 0.2) & np.isin(r['pct'], [0.1, 0.9]) | (r['alpha'] == 0.1) & np.isin(r['pct'], [0.25, 0.75]) & np.isin(np.abs(r['beta']), [0.5, 0.6, 0.7]) | (r['alpha'] == 0.1) & np.isin(r['pct'], [0.5]) & np.isin(np.abs(r['beta']), [0.1]) | (r['alpha'] == 0.1) & np.isin(r['pct'], [0.35, 0.65]) & np.isin(np.abs(r['beta']), [-0.4, -0.3, 0.3, 0.4, 0.5]) | (r['alpha'] == 0.2) & (r['beta'] == 0.5) & (r['pct'] == 0.25) | (r['alpha'] == 0.2) & (r['beta'] == -0.3) & (r['pct'] == 0.65) | (r['alpha'] == 0.2) & (r['beta'] == 0.3) & (r['pct'] == 0.35) | (r['alpha'] == 1.0) & np.isin(r['pct'], [0.5]) & np.isin(np.abs(r['beta']), [0.1, 0.2, 0.3, 0.4]) | (r['alpha'] == 1.0) & np.isin(r['pct'], [0.35, 0.65]) & np.isin(np.abs(r['beta']), [0.8, 0.9, 1.0]) | (r['alpha'] == 1.0) & np.isin(r['pct'], [0.01, 0.99]) & np.isin(np.abs(r['beta']), [-0.1, 0.1]) | (r['alpha'] >= 1.1))], ['piecewise', 1e-11, lambda r: np.isin(r['pct'], pct_range) & np.isin(r['alpha'], alpha_range) & np.isin(r['beta'], beta_range) & (r['alpha'] > 0.2) & (r['alpha'] != 1.0)], ['piecewise', 1e-11, lambda r: (r['alpha'] == 1.0) & (not is_linux_32) & np.isin(r['pct'], pct_range) & (1.0 in alpha_range) & np.isin(r['beta'], beta_range)], ['piecewise', 2.5e-10, lambda r: np.isin(r['pct'], pct_range) & np.isin(r['alpha'], alpha_range) & np.isin(r['beta'], beta_range) & (r['alpha'] <= 0.2)], ['fft-simpson', 1e-05, lambda r: (r['alpha'] >= 1.9) & np.isin(r['pct'], pct_range) & np.isin(r['alpha'], alpha_range) & np.isin(r['beta'], beta_range)], ['fft-simpson', 1e-06, lambda r: np.isin(r['pct'], pct_range) & np.isin(r['alpha'], alpha_range) & np.isin(r['beta'], beta_range) & (r['alpha'] > 1) & (r['alpha'] < 1.9)]]
        for ix, (default_method, rtol, filter_func) in enumerate(tests):
            stats.levy_stable.pdf_default_method = default_method
            subdata = data[filter_func(data)] if filter_func is not None else data
            with suppress_warnings() as sup:
                sup.record(RuntimeWarning, 'Density calculations experimental for FFT method.*')
                p = stats.levy_stable.pdf(subdata['x'], subdata['alpha'], subdata['beta'], scale=1, loc=0)
                with np.errstate(over='ignore'):
                    subdata2 = rec_append_fields(subdata, ['calc', 'abserr', 'relerr'], [p, np.abs(p - subdata['p']), np.abs(p - subdata['p']) / np.abs(subdata['p'])])
                failures = subdata2[(subdata2['relerr'] >= rtol) | np.isnan(p)]
                message = f"pdf test {ix} failed with method '{default_method}' [platform: {platform_desc}]\n{failures.dtype.names}\n{failures}"
                assert_allclose(p, subdata['p'], rtol, err_msg=message, verbose=False)

    @pytest.mark.parametrize('pct_range,alpha_range,beta_range', [pytest.param([0.01, 0.5, 0.99], [0.1, 1, 2], [-1, 0, 0.8]), pytest.param([0.01, 0.05, 0.5, 0.95, 0.99], [0.1, 0.5, 1, 1.5, 2], [-0.9, -0.5, 0, 0.3, 0.6, 1], marks=pytest.mark.slow), pytest.param([0.01, 0.05, 0.1, 0.25, 0.35, 0.5, 0.65, 0.75, 0.9, 0.95, 0.99], np.linspace(0.1, 2, 20), np.linspace(-1, 1, 21), marks=pytest.mark.xslow)])
    def test_cdf_nolan_samples(self, nolan_cdf_sample_data, pct_range, alpha_range, beta_range):
        """ Test cdf values against Nolan's stablec.exe output."""
        data = nolan_cdf_sample_data
        tests = [['piecewise', 2e-12, lambda r: np.isin(r['pct'], pct_range) & np.isin(r['alpha'], alpha_range) & np.isin(r['beta'], beta_range) & ~((r['alpha'] == 1.0) & np.isin(r['beta'], [-0.3, -0.2, -0.1]) & (r['pct'] == 0.01) | (r['alpha'] == 1.0) & np.isin(r['beta'], [0.1, 0.2, 0.3]) & (r['pct'] == 0.99))], ['piecewise', 0.05, lambda r: np.isin(r['pct'], pct_range) & np.isin(r['alpha'], alpha_range) & np.isin(r['beta'], beta_range) & ((r['alpha'] == 1.0) & np.isin(r['beta'], [-0.3, -0.2, -0.1]) & (r['pct'] == 0.01)) | (r['alpha'] == 1.0) & np.isin(r['beta'], [0.1, 0.2, 0.3]) & (r['pct'] == 0.99)], ['fft-simpson', 1e-05, lambda r: np.isin(r['pct'], pct_range) & np.isin(r['alpha'], alpha_range) & np.isin(r['beta'], beta_range) & (r['alpha'] > 1.7)], ['fft-simpson', 0.0001, lambda r: np.isin(r['pct'], pct_range) & np.isin(r['alpha'], alpha_range) & np.isin(r['beta'], beta_range) & (r['alpha'] > 1.5) & (r['alpha'] <= 1.7)], ['fft-simpson', 0.001, lambda r: np.isin(r['pct'], pct_range) & np.isin(r['alpha'], alpha_range) & np.isin(r['beta'], beta_range) & (r['alpha'] > 1.3) & (r['alpha'] <= 1.5)], ['fft-simpson', 0.01, lambda r: np.isin(r['pct'], pct_range) & np.isin(r['alpha'], alpha_range) & np.isin(r['beta'], beta_range) & (r['alpha'] > 1.0) & (r['alpha'] <= 1.3)]]
        for ix, (default_method, rtol, filter_func) in enumerate(tests):
            stats.levy_stable.cdf_default_method = default_method
            subdata = data[filter_func(data)] if filter_func is not None else data
            with suppress_warnings() as sup:
                sup.record(RuntimeWarning, 'Cumulative density calculations experimental for FFT' + ' method. Use piecewise method instead.*')
                p = stats.levy_stable.cdf(subdata['x'], subdata['alpha'], subdata['beta'], scale=1, loc=0)
                with np.errstate(over='ignore'):
                    subdata2 = rec_append_fields(subdata, ['calc', 'abserr', 'relerr'], [p, np.abs(p - subdata['p']), np.abs(p - subdata['p']) / np.abs(subdata['p'])])
                failures = subdata2[(subdata2['relerr'] >= rtol) | np.isnan(p)]
                message = f"cdf test {ix} failed with method '{default_method}'\n{failures.dtype.names}\n{failures}"
                assert_allclose(p, subdata['p'], rtol, err_msg=message, verbose=False)

    @pytest.mark.parametrize('param', [0, 1])
    @pytest.mark.parametrize('case', ['pdf', 'cdf'])
    def test_location_scale(self, nolan_loc_scale_sample_data, param, case):
        """Tests for pdf and cdf where loc, scale are different from 0, 1
        """
        uname = platform.uname()
        is_linux_32 = uname.system == 'Linux' and '32bit' in platform.architecture()[0]
        if is_linux_32 and case == 'pdf':
            pytest.skip('Test unstable on some platforms; see gh-17839, 17859')
        data = nolan_loc_scale_sample_data
        stats.levy_stable.cdf_default_method = 'piecewise'
        stats.levy_stable.pdf_default_method = 'piecewise'
        subdata = data[data['param'] == param]
        stats.levy_stable.parameterization = f'S{param}'
        assert case in ['pdf', 'cdf']
        function = stats.levy_stable.pdf if case == 'pdf' else stats.levy_stable.cdf
        v1 = function(subdata['x'], subdata['alpha'], subdata['beta'], scale=2, loc=3)
        assert_allclose(v1, subdata[case], 1e-05)

    @pytest.mark.parametrize('method,decimal_places', [['dni', 4], ['piecewise', 4]])
    def test_pdf_alpha_equals_one_beta_non_zero(self, method, decimal_places):
        """ sample points extracted from Tables and Graphs of Stable
        Probability Density Functions - Donald R Holt - 1973 - p 187.
        """
        xs = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
        density = np.array([0.3183, 0.3096, 0.2925, 0.2622, 0.1591, 0.1587, 0.1599, 0.1635, 0.0637, 0.0729, 0.0812, 0.0955, 0.0318, 0.039, 0.0458, 0.0586, 0.0187, 0.0236, 0.0285, 0.0384])
        betas = np.array([0, 0.25, 0.5, 1, 0, 0.25, 0.5, 1, 0, 0.25, 0.5, 1, 0, 0.25, 0.5, 1, 0, 0.25, 0.5, 1])
        with np.errstate(all='ignore'), suppress_warnings() as sup:
            sup.filter(category=RuntimeWarning, message='Density calculation unstable.*')
            stats.levy_stable.pdf_default_method = method
            pdf = stats.levy_stable.pdf(xs, 1, betas, scale=1, loc=0)
            assert_almost_equal(pdf, density, decimal_places, method)

    @pytest.mark.parametrize('params,expected', [[(1.48, -0.22, 0, 1), (0, np.inf, np.nan, np.nan)], [(2, 0.9, 10, 1.5), (10, 4.5, 0, 0)]])
    def test_stats(self, params, expected):
        observed = stats.levy_stable.stats(params[0], params[1], loc=params[2], scale=params[3], moments='mvsk')
        assert_almost_equal(observed, expected)

    @pytest.mark.parametrize('alpha', [0.25, 0.5, 0.75])
    @pytest.mark.parametrize('function,beta,points,expected', [(stats.levy_stable.cdf, 1.0, np.linspace(-25, 0, 10), 0.0), (stats.levy_stable.pdf, 1.0, np.linspace(-25, 0, 10), 0.0), (stats.levy_stable.cdf, -1.0, np.linspace(0, 25, 10), 1.0), (stats.levy_stable.pdf, -1.0, np.linspace(0, 25, 10), 0.0)])
    def test_distribution_outside_support(self, alpha, function, beta, points, expected):
        """Ensure the pdf/cdf routines do not return nan outside support.

        This distribution's support becomes truncated in a few special cases:
            support is [mu, infty) if alpha < 1 and beta = 1
            support is (-infty, mu] if alpha < 1 and beta = -1
        Otherwise, the support is all reals. Here, mu is zero by default.
        """
        assert 0 < alpha < 1
        assert_almost_equal(function(points, alpha=alpha, beta=beta), np.full(len(points), expected))

    @pytest.mark.parametrize('x,alpha,beta,expected', [(0, 1.7720732804618808, 0.5059373136902996, 0.278932636798268), (0, 1.9217001522410235, -0.8779442746685926, 0.281054757202316), (0, 1.5654806051633634, -0.4016220341911392, 0.271282133194204), (0, 1.7420803447784388, -0.38180029468259247, 0.280202199244247), (0, 1.5748002527689913, -0.25200194914153684, 0.280136576218665)])
    def test_x_equal_zeta(self, x, alpha, beta, expected):
        """Test pdf for x equal to zeta.

        With S1 parametrization: x0 = x + zeta if alpha != 1 So, for x = 0, x0
        will be close to zeta.

        When case "x equal zeta" is not handled properly and quad_eps is not
        low enough: - pdf may be less than 0 - logpdf is nan

        The points from the parametrize block are found randomly so that PDF is
        less than 0.

        Reference values taken from MATLAB
        https://www.mathworks.com/help/stats/stable-distribution.html
        """
        stats.levy_stable.quad_eps = 1.2e-11
        assert_almost_equal(stats.levy_stable.pdf(x, alpha=alpha, beta=beta), expected)

    @pytest.mark.xfail
    @pytest.mark.parametrize('x,alpha,beta,expected', [(0.0001, 1.7720732804618808, 0.5059373136902996, 0.27892916534067), (0.0001, 1.9217001522410235, -0.8779442746685926, 0.281056564327953), (0.0001, 1.5654806051633634, -0.4016220341911392, 0.271252432161167), (0.0001, 1.7420803447784388, -0.38180029468259247, 0.280205311264134), (0.0001, 1.5748002527689913, -0.25200194914153684, 0.280140965235426), (-0.0001, 1.7720732804618808, 0.5059373136902996, 0.278936106741754), (-0.0001, 1.9217001522410235, -0.8779442746685926, 0.281052948629429), (-0.0001, 1.5654806051633634, -0.4016220341911392, 0.271275394392385), (-0.0001, 1.7420803447784388, -0.38180029468259247, 0.280199085645099), (-0.0001, 1.5748002527689913, -0.25200194914153684, 0.280132185432842)])
    def test_x_near_zeta(self, x, alpha, beta, expected):
        """Test pdf for x near zeta.

        With S1 parametrization: x0 = x + zeta if alpha != 1 So, for x = 0, x0
        will be close to zeta.

        When case "x near zeta" is not handled properly and quad_eps is not
        low enough: - pdf may be less than 0 - logpdf is nan

        The points from the parametrize block are found randomly so that PDF is
        less than 0.

        Reference values taken from MATLAB
        https://www.mathworks.com/help/stats/stable-distribution.html
        """
        stats.levy_stable.quad_eps = 1.2e-11
        assert_almost_equal(stats.levy_stable.pdf(x, alpha=alpha, beta=beta), expected)