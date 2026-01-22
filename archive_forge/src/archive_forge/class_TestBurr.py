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
class TestBurr:

    def test_endpoints_7491(self):
        data = [[stats.fisk, (1,), 1], [stats.burr, (0.5, 2), 1], [stats.burr, (1, 1), 1], [stats.burr, (2, 0.5), 1], [stats.burr12, (1, 0.5), 0.5], [stats.burr12, (1, 1), 1.0], [stats.burr12, (1, 2), 2.0]]
        ans = [_f.pdf(_f.a, *_args) for _f, _args, _ in data]
        correct = [_correct_ for _f, _args, _correct_ in data]
        assert_array_almost_equal(ans, correct)
        ans = [_f.logpdf(_f.a, *_args) for _f, _args, _ in data]
        correct = [np.log(_correct_) for _f, _args, _correct_ in data]
        assert_array_almost_equal(ans, correct)

    def test_burr_stats_9544(self):
        c, d = (5.0, 3)
        mean, variance = stats.burr(c, d).stats()
        mean_hc, variance_hc = (1.4110263183925857, 0.22879948026191643)
        assert_allclose(mean, mean_hc)
        assert_allclose(variance, variance_hc)

    def test_burr_nan_mean_var_9544(self):
        c, d = (0.5, 3)
        mean, variance = stats.burr(c, d).stats()
        assert_(np.isnan(mean))
        assert_(np.isnan(variance))
        c, d = (1.5, 3)
        mean, variance = stats.burr(c, d).stats()
        assert_(np.isfinite(mean))
        assert_(np.isnan(variance))
        c, d = (0.5, 3)
        e1, e2, e3, e4 = stats.burr._munp(np.array([1, 2, 3, 4]), c, d)
        assert_(np.isnan(e1))
        assert_(np.isnan(e2))
        assert_(np.isnan(e3))
        assert_(np.isnan(e4))
        c, d = (1.5, 3)
        e1, e2, e3, e4 = stats.burr._munp([1, 2, 3, 4], c, d)
        assert_(np.isfinite(e1))
        assert_(np.isnan(e2))
        assert_(np.isnan(e3))
        assert_(np.isnan(e4))
        c, d = (2.5, 3)
        e1, e2, e3, e4 = stats.burr._munp([1, 2, 3, 4], c, d)
        assert_(np.isfinite(e1))
        assert_(np.isfinite(e2))
        assert_(np.isnan(e3))
        assert_(np.isnan(e4))
        c, d = (3.5, 3)
        e1, e2, e3, e4 = stats.burr._munp([1, 2, 3, 4], c, d)
        assert_(np.isfinite(e1))
        assert_(np.isfinite(e2))
        assert_(np.isfinite(e3))
        assert_(np.isnan(e4))
        c, d = (4.5, 3)
        e1, e2, e3, e4 = stats.burr._munp([1, 2, 3, 4], c, d)
        assert_(np.isfinite(e1))
        assert_(np.isfinite(e2))
        assert_(np.isfinite(e3))
        assert_(np.isfinite(e4))

    def test_burr_isf(self):
        c, d = (5.0, 3.0)
        q = [0.1, 1e-10, 1e-20, 1e-40]
        ref = [1.9469686558286508, 124.57309395989076, 12457.309396155173, 124573093.96155174]
        assert_allclose(stats.burr.isf(q, c, d), ref, rtol=1e-14)