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
class TestDiscreteAliasUrn:
    basic_fail_dists = {'nchypergeom_fisher', 'nchypergeom_wallenius', 'randint'}

    @pytest.mark.parametrize('distname, params', distdiscrete)
    def test_basic(self, distname, params):
        if distname in self.basic_fail_dists:
            msg = 'DAU fails on these probably because of large domains and small computation errors in PMF.'
            pytest.skip(msg)
        if not isinstance(distname, str):
            dist = distname
        else:
            dist = getattr(stats, distname)
        dist = dist(*params)
        domain = dist.support()
        if not np.isfinite(domain[1] - domain[0]):
            pytest.skip('DAU only works with a finite domain.')
        k = np.arange(domain[0], domain[1] + 1)
        pv = dist.pmf(k)
        mv_ex = dist.stats('mv')
        rng = DiscreteAliasUrn(dist, random_state=42)
        check_discr_samples(rng, pv, mv_ex)
    bad_pmf = [(lambda x: np.inf, ValueError, 'must contain only finite / non-nan values'), (lambda x: np.nan, ValueError, 'must contain only finite / non-nan values'), (lambda x: 0.0, ValueError, 'must contain at least one non-zero value'), (lambda x: foo, NameError, "name 'foo' is not defined"), (lambda x: [], ValueError, 'setting an array element with a sequence.'), (lambda x: -x, UNURANError, '50 : probability < 0'), (lambda: 1.0, TypeError, 'takes 0 positional arguments but 1 was given')]

    @pytest.mark.parametrize('pmf, err, msg', bad_pmf)
    def test_bad_pmf(self, pmf, err, msg):

        class dist:
            pass
        dist.pmf = pmf
        with pytest.raises(err, match=msg):
            DiscreteAliasUrn(dist, domain=(1, 10))

    @pytest.mark.parametrize('pv', [[0.18, 0.02, 0.8], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    def test_sampling_with_pv(self, pv):
        pv = np.asarray(pv, dtype=np.float64)
        rng = DiscreteAliasUrn(pv, random_state=123)
        rng.rvs(100000)
        pv = pv / pv.sum()
        variates = np.arange(0, len(pv))
        m_expected = np.average(variates, weights=pv)
        v_expected = np.average((variates - m_expected) ** 2, weights=pv)
        mv_expected = (m_expected, v_expected)
        check_discr_samples(rng, pv, mv_expected)

    @pytest.mark.parametrize('pv, msg', bad_pv_common)
    def test_bad_pv(self, pv, msg):
        with pytest.raises(ValueError, match=msg):
            DiscreteAliasUrn(pv)
    inf_domain = [(-np.inf, np.inf), (np.inf, np.inf), (-np.inf, -np.inf), (0, np.inf), (-np.inf, 0)]

    @pytest.mark.parametrize('domain', inf_domain)
    def test_inf_domain(self, domain):
        with pytest.raises(ValueError, match='must be finite'):
            DiscreteAliasUrn(stats.binom(10, 0.2), domain=domain)

    def test_bad_urn_factor(self):
        with pytest.warns(RuntimeWarning, match='relative urn size < 1.'):
            DiscreteAliasUrn([0.5, 0.5], urn_factor=-1)

    def test_bad_args(self):
        msg = '`domain` must be provided when the probability vector is not available.'

        class dist:

            def pmf(self, x):
                return x
        with pytest.raises(ValueError, match=msg):
            DiscreteAliasUrn(dist)

    def test_gh19359(self):
        pv = special.softmax(np.ones((1533,)))
        rng = DiscreteAliasUrn(pv, random_state=42)
        check_discr_samples(rng, pv, (1532 / 2, (1532 ** 2 - 1) / 12), rtol=0.005)