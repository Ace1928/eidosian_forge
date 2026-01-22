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
class TestRelativisticBW:

    @pytest.fixture
    def ROOT_pdf_sample_data(self):
        """Sample data points for pdf computed with CERN's ROOT

        See - https://root.cern/

        Uses ROOT.TMath.BreitWignerRelativistic, available in ROOT
        versions 6.27+

        pdf calculated for Z0 Boson, W Boson, and Higgs Boson for
        x in `np.linspace(0, 200, 401)`.
        """
        data = np.load(Path(__file__).parent / 'data/rel_breitwigner_pdf_sample_data_ROOT.npy')
        data = np.rec.fromarrays(data.T, names='x,pdf,rho,gamma')
        return data

    @pytest.mark.parametrize('rho,gamma,rtol', [(36.545206797050334, 2.4952, 5e-14), (38.55107913669065, 2.085, 1e-14), (96292.3076923077, 0.0013, 5e-13)])
    def test_pdf_against_ROOT(self, ROOT_pdf_sample_data, rho, gamma, rtol):
        data = ROOT_pdf_sample_data[(ROOT_pdf_sample_data['rho'] == rho) & (ROOT_pdf_sample_data['gamma'] == gamma)]
        x, pdf = (data['x'], data['pdf'])
        assert_allclose(pdf, stats.rel_breitwigner.pdf(x, rho, scale=gamma), rtol=rtol)

    @pytest.mark.parametrize('rho, Gamma, rtol', [(36.545206797050334, 2.4952, 5e-13), (38.55107913669065, 2.085, 5e-13), (96292.3076923077, 0.0013, 5e-10)])
    def test_pdf_against_simple_implementation(self, rho, Gamma, rtol):

        def pdf(E, M, Gamma):
            gamma = np.sqrt(M ** 2 * (M ** 2 + Gamma ** 2))
            k = 2 * np.sqrt(2) * M * Gamma * gamma / (np.pi * np.sqrt(M ** 2 + gamma))
            return k / ((E ** 2 - M ** 2) ** 2 + M ** 2 * Gamma ** 2)
        p = np.linspace(0.05, 0.95, 10)
        x = stats.rel_breitwigner.ppf(p, rho, scale=Gamma)
        res = stats.rel_breitwigner.pdf(x, rho, scale=Gamma)
        ref = pdf(x, rho * Gamma, Gamma)
        assert_allclose(res, ref, rtol=rtol)

    @pytest.mark.parametrize('rho,gamma', [pytest.param(36.545206797050334, 2.4952, marks=pytest.mark.slow), pytest.param(38.55107913669065, 2.085, marks=pytest.mark.xslow), pytest.param(96292.3076923077, 0.0013, marks=pytest.mark.xslow)])
    def test_fit_floc(self, rho, gamma):
        """Tests fit for cases where floc is set.

        `rel_breitwigner` has special handling for these cases.
        """
        seed = 6936804688480013683
        rng = np.random.default_rng(seed)
        data = stats.rel_breitwigner.rvs(rho, scale=gamma, size=1000, random_state=rng)
        fit = stats.rel_breitwigner.fit(data, floc=0)
        assert_allclose((fit[0], fit[2]), (rho, gamma), rtol=0.2)
        assert fit[1] == 0
        fit = stats.rel_breitwigner.fit(data, floc=0, fscale=gamma)
        assert_allclose(fit[0], rho, rtol=0.01)
        assert (fit[1], fit[2]) == (0, gamma)