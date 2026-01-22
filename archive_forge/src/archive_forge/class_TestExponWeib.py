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
class TestExponWeib:

    def test_pdf_logpdf(self):
        x = 0.1
        a = 1.0
        c = 100.0
        p = stats.exponweib.pdf(x, a, c)
        logp = stats.exponweib.logpdf(x, a, c)
        assert_allclose([p, logp], [1.0000000000000054e-97, -223.35075402042244])

    def test_a_is_1(self):
        x = np.logspace(-4, -1, 4)
        a = 1
        c = 100
        p = stats.exponweib.pdf(x, a, c)
        expected = stats.weibull_min.pdf(x, c)
        assert_allclose(p, expected)
        logp = stats.exponweib.logpdf(x, a, c)
        expected = stats.weibull_min.logpdf(x, c)
        assert_allclose(logp, expected)

    def test_a_is_1_c_is_1(self):
        x = np.logspace(-8, 1, 10)
        a = 1
        c = 1
        p = stats.exponweib.pdf(x, a, c)
        expected = stats.expon.pdf(x)
        assert_allclose(p, expected)
        logp = stats.exponweib.logpdf(x, a, c)
        expected = stats.expon.logpdf(x)
        assert_allclose(logp, expected)

    @pytest.mark.parametrize('x, a, c, ref', [(1, 2.5, 0.75, 0.6823127476985246), (50, 2.5, 0.75, 1.7056666054719663e-08), (125, 2.5, 0.75, 1.4534393150714602e-16), (250, 2.5, 0.75, 1.2391389689773512e-27), (250, 0.03125, 0.75, 1.548923711221689e-29), (3, 0.03125, 3.0, 5.873527551689983e-14), (2e+80, 10.0, 0.02, 2.9449084156902135e-17)])
    def test_sf(self, x, a, c, ref):
        sf = stats.exponweib.sf(x, a, c)
        assert_allclose(sf, ref, rtol=1e-14)

    @pytest.mark.parametrize('p, a, c, ref', [(0.25, 2.5, 0.75, 2.8946008178158924), (3e-16, 2.5, 0.75, 121.77966713102938), (1e-12, 1, 2, 5.256521769756932), (2e-13, 0.03125, 3, 2.953915059484589), (5e-14, 10.0, 0.02, 7.57094886384687e+75)])
    def test_isf(self, p, a, c, ref):
        isf = stats.exponweib.isf(p, a, c)
        assert_allclose(isf, ref, rtol=5e-14)