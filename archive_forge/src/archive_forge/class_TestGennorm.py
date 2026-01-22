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
class TestGennorm:

    def test_laplace(self):
        points = [1, 2, 3]
        pdf1 = stats.gennorm.pdf(points, 1)
        pdf2 = stats.laplace.pdf(points)
        assert_almost_equal(pdf1, pdf2)

    def test_norm(self):
        points = [1, 2, 3]
        pdf1 = stats.gennorm.pdf(points, 2)
        pdf2 = stats.norm.pdf(points, scale=2 ** (-0.5))
        assert_almost_equal(pdf1, pdf2)

    def test_rvs(self):
        np.random.seed(0)
        dist = stats.gennorm(0.5)
        rvs = dist.rvs(size=1000)
        assert stats.kstest(rvs, dist.cdf).pvalue > 0.1
        dist = stats.gennorm(1)
        rvs = dist.rvs(size=1000)
        rvs_laplace = stats.laplace.rvs(size=1000)
        assert stats.ks_2samp(rvs, rvs_laplace).pvalue > 0.1
        dist = stats.gennorm(2)
        rvs = dist.rvs(size=1000)
        rvs_norm = stats.norm.rvs(scale=1 / 2 ** 0.5, size=1000)
        assert stats.ks_2samp(rvs, rvs_norm).pvalue > 0.1

    def test_rvs_broadcasting(self):
        np.random.seed(0)
        dist = stats.gennorm([[0.5, 1.0], [2.0, 5.0]])
        rvs = dist.rvs(size=[1000, 2, 2])
        assert stats.kstest(rvs[:, 0, 0], stats.gennorm(0.5).cdf)[1] > 0.1
        assert stats.kstest(rvs[:, 0, 1], stats.gennorm(1.0).cdf)[1] > 0.1
        assert stats.kstest(rvs[:, 1, 0], stats.gennorm(2.0).cdf)[1] > 0.1
        assert stats.kstest(rvs[:, 1, 1], stats.gennorm(5.0).cdf)[1] > 0.1