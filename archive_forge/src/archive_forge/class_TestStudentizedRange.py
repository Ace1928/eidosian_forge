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
class TestStudentizedRange:
    q05 = [17.97, 45.4, 54.33, 59.56, 4.501, 8.853, 10.35, 11.24, 3.151, 5.305, 6.028, 6.467, 2.95, 4.768, 5.357, 5.714, 2.8, 4.363, 4.842, 5.126, 2.772, 4.286, 4.743, 5.012]
    q01 = [90.03, 227.2, 271.8, 298.0, 8.261, 15.64, 18.22, 19.77, 4.482, 6.875, 7.712, 8.226, 4.024, 5.839, 6.45, 6.823, 3.702, 5.118, 5.562, 5.827, 3.643, 4.987, 5.4, 5.645]
    q001 = [900.3, 2272, 2718, 2980, 18.28, 34.12, 39.69, 43.05, 6.487, 9.352, 10.39, 11.03, 5.444, 7.313, 7.966, 8.37, 4.772, 6.039, 6.448, 6.695, 4.654, 5.823, 6.191, 6.411]
    qs = np.concatenate((q05, q01, q001))
    ps = [0.95, 0.99, 0.999]
    vs = [1, 3, 10, 20, 120, np.inf]
    ks = [2, 8, 14, 20]
    data = list(zip(product(ps, vs, ks), qs))
    r_data = [(0.1, 3, 9001, 0.002752818526842), (1, 10, 1000, 0.000526142388912), (1, 3, np.inf, 0.240712641229283), (4, 3, np.inf, 0.987012338626815), (1, 10, np.inf, 0.000519869467083)]

    def test_cdf_against_tables(self):
        for pvk, q in self.data:
            p_expected, v, k = pvk
            res_p = stats.studentized_range.cdf(q, k, v)
            assert_allclose(res_p, p_expected, rtol=0.0001)

    @pytest.mark.slow
    def test_ppf_against_tables(self):
        for pvk, q_expected in self.data:
            p, v, k = pvk
            res_q = stats.studentized_range.ppf(p, k, v)
            assert_allclose(res_q, q_expected, rtol=0.0005)
    path_prefix = os.path.dirname(__file__)
    relative_path = 'data/studentized_range_mpmath_ref.json'
    with open(os.path.join(path_prefix, relative_path)) as file:
        pregenerated_data = json.load(file)

    @pytest.mark.parametrize('case_result', pregenerated_data['cdf_data'])
    def test_cdf_against_mp(self, case_result):
        src_case = case_result['src_case']
        mp_result = case_result['mp_result']
        qkv = (src_case['q'], src_case['k'], src_case['v'])
        res = stats.studentized_range.cdf(*qkv)
        assert_allclose(res, mp_result, atol=src_case['expected_atol'], rtol=src_case['expected_rtol'])

    @pytest.mark.parametrize('case_result', pregenerated_data['pdf_data'])
    def test_pdf_against_mp(self, case_result):
        src_case = case_result['src_case']
        mp_result = case_result['mp_result']
        qkv = (src_case['q'], src_case['k'], src_case['v'])
        res = stats.studentized_range.pdf(*qkv)
        assert_allclose(res, mp_result, atol=src_case['expected_atol'], rtol=src_case['expected_rtol'])

    @pytest.mark.slow
    @pytest.mark.xfail_on_32bit('intermittent RuntimeWarning: invalid value.')
    @pytest.mark.parametrize('case_result', pregenerated_data['moment_data'])
    def test_moment_against_mp(self, case_result):
        src_case = case_result['src_case']
        mp_result = case_result['mp_result']
        mkv = (src_case['m'], src_case['k'], src_case['v'])
        with np.errstate(invalid='ignore'):
            res = stats.studentized_range.moment(*mkv)
        assert_allclose(res, mp_result, atol=src_case['expected_atol'], rtol=src_case['expected_rtol'])

    def test_pdf_integration(self):
        k, v = (3, 10)
        res = quad(stats.studentized_range.pdf, 0, np.inf, args=(k, v))
        assert_allclose(res[0], 1)

    @pytest.mark.xslow
    def test_pdf_against_cdf(self):
        k, v = (3, 10)
        x = np.arange(0, 10, step=0.01)
        y_cdf = stats.studentized_range.cdf(x, k, v)[1:]
        y_pdf_raw = stats.studentized_range.pdf(x, k, v)
        y_pdf_cumulative = cumulative_trapezoid(y_pdf_raw, x)
        assert_allclose(y_pdf_cumulative, y_cdf, rtol=0.0001)

    @pytest.mark.parametrize('r_case_result', r_data)
    def test_cdf_against_r(self, r_case_result):
        q, k, v, r_res = r_case_result
        with np.errstate(invalid='ignore'):
            res = stats.studentized_range.cdf(q, k, v)
        assert_allclose(res, r_res)

    @pytest.mark.slow
    @pytest.mark.xfail_on_32bit('intermittent RuntimeWarning: invalid value.')
    def test_moment_vectorization(self):
        with np.errstate(invalid='ignore'):
            m = stats.studentized_range._munp([1, 2], [4, 5], [10, 11])
        assert_allclose(m.shape, (2,))
        with pytest.raises(ValueError, match='...could not be broadcast...'):
            stats.studentized_range._munp(1, [4, 5], [10, 11, 12])

    @pytest.mark.xslow
    def test_fitstart_valid(self):
        with suppress_warnings() as sup, np.errstate(invalid='ignore'):
            sup.filter(IntegrationWarning)
            k, df, _, _ = stats.studentized_range._fitstart([1, 2, 3])
        assert_(stats.studentized_range._argcheck(k, df))

    def test_infinite_df(self):
        res = stats.studentized_range.pdf(3, 10, np.inf)
        res_finite = stats.studentized_range.pdf(3, 10, 99999)
        assert_allclose(res, res_finite, atol=0.0001, rtol=0.0001)
        res = stats.studentized_range.cdf(3, 10, np.inf)
        res_finite = stats.studentized_range.cdf(3, 10, 99999)
        assert_allclose(res, res_finite, atol=0.0001, rtol=0.0001)

    def test_df_cutoff(self):
        res = stats.studentized_range.pdf(3, 10, 100000)
        res_finite = stats.studentized_range.pdf(3, 10, 99999)
        res_sanity = stats.studentized_range.pdf(3, 10, 99998)
        assert_raises(AssertionError, assert_allclose, res, res_finite, atol=1e-06, rtol=1e-06)
        assert_allclose(res_finite, res_sanity, atol=1e-06, rtol=1e-06)
        res = stats.studentized_range.cdf(3, 10, 100000)
        res_finite = stats.studentized_range.cdf(3, 10, 99999)
        res_sanity = stats.studentized_range.cdf(3, 10, 99998)
        assert_raises(AssertionError, assert_allclose, res, res_finite, atol=1e-06, rtol=1e-06)
        assert_allclose(res_finite, res_sanity, atol=1e-06, rtol=1e-06)

    def test_clipping(self):
        q, k, v = (34.641399619534575, 3, 339)
        p = stats.studentized_range.sf(q, k, v)
        assert_allclose(p, 0, atol=1e-10)
        assert p >= 0