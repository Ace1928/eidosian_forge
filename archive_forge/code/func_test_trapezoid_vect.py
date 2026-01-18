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
def test_trapezoid_vect(self):
    c = np.array([0.1, 0.2, 0.3])
    d = np.array([0.5, 0.6])[:, None]
    x = np.array([0.15, 0.25, 0.9])
    v = stats.trapezoid.pdf(x, c, d)
    cc, dd, xx = np.broadcast_arrays(c, d, x)
    res = np.empty(xx.size, dtype=xx.dtype)
    ind = np.arange(xx.size)
    for i, x1, c1, d1 in zip(ind, xx.ravel(), cc.ravel(), dd.ravel()):
        res[i] = stats.trapezoid.pdf(x1, c1, d1)
    assert_allclose(v, res.reshape(v.shape), atol=1e-15)
    v = np.asarray(stats.trapezoid.stats(c, d, moments='mvsk'))
    cc, dd = np.broadcast_arrays(c, d)
    res = np.empty((cc.size, 4))
    ind = np.arange(cc.size)
    for i, c1, d1 in zip(ind, cc.ravel(), dd.ravel()):
        res[i] = stats.trapezoid.stats(c1, d1, moments='mvsk')
    assert_allclose(v, res.T.reshape(v.shape), atol=1e-15)