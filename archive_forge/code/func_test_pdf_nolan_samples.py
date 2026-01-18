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