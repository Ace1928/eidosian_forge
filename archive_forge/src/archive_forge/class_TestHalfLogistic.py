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
class TestHalfLogistic:

    @pytest.mark.parametrize('x, ref', [(100, 7.440151952041672e-44), (200, 2.767793053473475e-87)])
    def test_sf(self, x, ref):
        assert_allclose(stats.halflogistic.sf(x), ref, rtol=1e-15)

    @pytest.mark.parametrize('q, ref', [(7.440151952041672e-44, 100), (2.767793053473475e-87, 200), (1 - 1e-09, 1.999999943436137e-09), (1 - 1e-15, 1.9984014443252818e-15)])
    def test_isf(self, q, ref):
        assert_allclose(stats.halflogistic.isf(q), ref, rtol=1e-15)

    @pytest.mark.parametrize('rvs_loc', [1e-05, 10000000000.0])
    @pytest.mark.parametrize('rvs_scale', [0.01, 100, 100000000.0])
    @pytest.mark.parametrize('fix_loc', [True, False])
    @pytest.mark.parametrize('fix_scale', [True, False])
    def test_fit_MLE_comp_optimizer(self, rvs_loc, rvs_scale, fix_loc, fix_scale):
        rng = np.random.default_rng(6762668991392531563)
        data = stats.halflogistic.rvs(loc=rvs_loc, scale=rvs_scale, size=1000, random_state=rng)
        kwds = {}
        if fix_loc and fix_scale:
            error_msg = 'All parameters fixed. There is nothing to optimize.'
            with pytest.raises(RuntimeError, match=error_msg):
                stats.halflogistic.fit(data, floc=rvs_loc, fscale=rvs_scale)
            return
        if fix_loc:
            kwds['floc'] = rvs_loc
        if fix_scale:
            kwds['fscale'] = rvs_scale
        _assert_less_or_close_loglike(stats.halflogistic, data, **kwds)

    def test_fit_bad_floc(self):
        msg = " Maximum likelihood estimation with 'halflogistic' requires"
        with assert_raises(FitDataError, match=msg):
            stats.halflogistic.fit([0, 2, 4], floc=1)