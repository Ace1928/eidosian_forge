import threading
import pickle
import pytest
from copy import deepcopy
import platform
import sys
import math
import numpy as np
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy.stats.sampling import (
from pytest import raises as assert_raises
from scipy import stats
from scipy import special
from scipy.stats import chisquare, cramervonmises
from scipy.stats._distr_params import distdiscrete, distcont
from scipy._lib._util import check_random_state
class TestTransformedDensityRejection:

    class dist0:

        def pdf(self, x):
            return 3 / 4 * (1 - x * x)

        def dpdf(self, x):
            return 3 / 4 * (-2 * x)

        def cdf(self, x):
            return 3 / 4 * (x - x ** 3 / 3 + 2 / 3)

        def support(self):
            return (-1, 1)

    class dist1:

        def pdf(self, x):
            return stats.norm._pdf(x / 0.1)

        def dpdf(self, x):
            return -x / 0.01 * stats.norm._pdf(x / 0.1)

        def cdf(self, x):
            return stats.norm._cdf(x / 0.1)

    class dist2:

        def __init__(self, shift):
            self.shift = shift

        def pdf(self, x):
            x -= self.shift
            y = 1.0 / (abs(x) + 1.0)
            return 0.5 * y * y

        def dpdf(self, x):
            x -= self.shift
            y = 1.0 / (abs(x) + 1.0)
            y = y * y * y
            return y if x < 0.0 else -y

        def cdf(self, x):
            x -= self.shift
            if x <= 0.0:
                return 0.5 / (1.0 - x)
            else:
                return 1.0 - 0.5 / (1.0 + x)
    dists = [dist0(), dist1(), dist2(0.0), dist2(10000.0)]
    mv0 = [0.0, 4.0 / 15.0]
    mv1 = [0.0, 0.01]
    mv2 = [0.0, np.inf]
    mv3 = [10000.0, np.inf]
    mvs = [mv0, mv1, mv2, mv3]

    @pytest.mark.parametrize('dist, mv_ex', zip(dists, mvs))
    def test_basic(self, dist, mv_ex):
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            rng = TransformedDensityRejection(dist, random_state=42)
        check_cont_samples(rng, dist, mv_ex)
    bad_pdfs = [(lambda x: 0, UNURANError, '50 : bad construction points.')]
    bad_pdfs += bad_pdfs_common

    @pytest.mark.parametrize('pdf, err, msg', bad_pdfs)
    def test_bad_pdf(self, pdf, err, msg):

        class dist:
            pass
        dist.pdf = pdf
        dist.dpdf = lambda x: 1
        with pytest.raises(err, match=msg):
            TransformedDensityRejection(dist)

    @pytest.mark.parametrize('dpdf, err, msg', bad_dpdf_common)
    def test_bad_dpdf(self, dpdf, err, msg):

        class dist:
            pass
        dist.pdf = lambda x: x
        dist.dpdf = dpdf
        with pytest.raises(err, match=msg):
            TransformedDensityRejection(dist, domain=(1, 10))

    @pytest.mark.parametrize('domain, err, msg', inf_nan_domains)
    def test_inf_nan_domains(self, domain, err, msg):
        with pytest.raises(err, match=msg):
            TransformedDensityRejection(StandardNormal(), domain=domain)

    @pytest.mark.parametrize('construction_points', [-1, 0, 0.1])
    def test_bad_construction_points_scalar(self, construction_points):
        with pytest.raises(ValueError, match='`construction_points` must be a positive integer.'):
            TransformedDensityRejection(StandardNormal(), construction_points=construction_points)

    def test_bad_construction_points_array(self):
        construction_points = []
        with pytest.raises(ValueError, match='`construction_points` must either be a scalar or a non-empty array.'):
            TransformedDensityRejection(StandardNormal(), construction_points=construction_points)
        construction_points = [1, 1, 1, 1, 1, 1]
        with pytest.warns(RuntimeWarning, match='33 : starting points not strictly monotonically increasing'):
            TransformedDensityRejection(StandardNormal(), construction_points=construction_points)
        construction_points = [np.nan, np.nan, np.nan]
        with pytest.raises(UNURANError, match='50 : bad construction points.'):
            TransformedDensityRejection(StandardNormal(), construction_points=construction_points)
        construction_points = [-10, 10]
        with pytest.warns(RuntimeWarning, match='50 : starting point out of domain'):
            TransformedDensityRejection(StandardNormal(), domain=(-3, 3), construction_points=construction_points)

    @pytest.mark.parametrize('c', [-1.0, np.nan, np.inf, 0.1, 1.0])
    def test_bad_c(self, c):
        msg = '`c` must either be -0.5 or 0.'
        with pytest.raises(ValueError, match=msg):
            TransformedDensityRejection(StandardNormal(), c=-1.0)
    u = [np.linspace(0, 1, num=1000), [], [[]], [np.nan], [-np.inf, np.nan, np.inf], 0, [[np.nan, 0.5, 0.1], [0.2, 0.4, np.inf], [-2, 3, 4]]]

    @pytest.mark.parametrize('u', u)
    def test_ppf_hat(self, u):
        rng = TransformedDensityRejection(StandardNormal(), max_squeeze_hat_ratio=0.9999)
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, 'invalid value encountered in greater')
            sup.filter(RuntimeWarning, 'invalid value encountered in greater_equal')
            sup.filter(RuntimeWarning, 'invalid value encountered in less')
            sup.filter(RuntimeWarning, 'invalid value encountered in less_equal')
            res = rng.ppf_hat(u)
            expected = stats.norm.ppf(u)
        assert_allclose(res, expected, rtol=0.001, atol=1e-05)
        assert res.shape == expected.shape

    def test_bad_dist(self):

        class dist:
            ...
        msg = '`pdf` required but not found.'
        with pytest.raises(ValueError, match=msg):
            TransformedDensityRejection(dist)

        class dist:
            pdf = lambda x: 1 - x * x
        msg = '`dpdf` required but not found.'
        with pytest.raises(ValueError, match=msg):
            TransformedDensityRejection(dist)