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
@pytest.mark.parametrize('methname', ['pdf', 'logpdf', 'cdf', 'ppf', 'sf', 'isf'])
@pytest.mark.parametrize('df_infmask', [[0, 0], [1, 1], [0, 1], [[0, 1, 0], [1, 1, 1]], [[1, 0], [0, 1]], [[0], [1]]])
def test_t_inf_df(self, methname, df_infmask):
    np.random.seed(0)
    df_infmask = np.asarray(df_infmask, dtype=bool)
    df = np.random.uniform(0, 10, size=df_infmask.shape)
    x = np.random.randn(*df_infmask.shape)
    df[df_infmask] = np.inf
    t_dist = stats.t(df=df, loc=3, scale=1)
    t_dist_ref = stats.t(df=df[~df_infmask], loc=3, scale=1)
    norm_dist = stats.norm(loc=3, scale=1)
    t_meth = getattr(t_dist, methname)
    t_meth_ref = getattr(t_dist_ref, methname)
    norm_meth = getattr(norm_dist, methname)
    res = t_meth(x)
    assert_equal(res[df_infmask], norm_meth(x[df_infmask]))
    assert_equal(res[~df_infmask], t_meth_ref(x[~df_infmask]))