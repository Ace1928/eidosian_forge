import os
import re
import warnings
from collections import namedtuple
from itertools import product
import hypothesis.extra.numpy as npst
import hypothesis
import contextlib
from numpy.testing import (assert_, assert_equal,
import pytest
from pytest import raises as assert_raises
import numpy.ma.testutils as mat
from numpy import array, arange, float32, float64, power
import numpy as np
import scipy.stats as stats
import scipy.stats.mstats as mstats
import scipy.stats._mstats_basic as mstats_basic
from scipy.stats._ksstats import kolmogn
from scipy.special._testutils import FuncData
from scipy.special import binom
from scipy import optimize
from .common_tests import check_named_results
from scipy.spatial.distance import cdist
from scipy.stats._axis_nan_policy import _broadcast_concatenate
from scipy.stats._stats_py import _permutation_distribution_t
from scipy._lib._util import AxisError
def _simulations(self, samps=100, dims=1, sim_type=''):
    if sim_type == 'linear':
        x = np.random.uniform(-1, 1, size=(samps, 1))
        y = x + 0.3 * np.random.random_sample(size=(x.size, 1))
    elif sim_type == 'nonlinear':
        unif = np.array(np.random.uniform(0, 5, size=(samps, 1)))
        x = unif * np.cos(np.pi * unif)
        y = unif * np.sin(np.pi * unif) + 0.4 * np.random.random_sample(size=(x.size, 1))
    elif sim_type == 'independence':
        u = np.random.normal(0, 1, size=(samps, 1))
        v = np.random.normal(0, 1, size=(samps, 1))
        u_2 = np.random.binomial(1, p=0.5, size=(samps, 1))
        v_2 = np.random.binomial(1, p=0.5, size=(samps, 1))
        x = u / 3 + 2 * u_2 - 1
        y = v / 3 + 2 * v_2 - 1
    else:
        raise ValueError('sim_type must be linear, nonlinear, or independence')
    if dims > 1:
        dims_noise = np.random.normal(0, 1, size=(samps, dims - 1))
        x = np.concatenate((x, dims_noise), axis=1)
    return (x, y)