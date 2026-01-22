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
class TestHalfgennorm:

    def test_expon(self):
        points = [1, 2, 3]
        pdf1 = stats.halfgennorm.pdf(points, 1)
        pdf2 = stats.expon.pdf(points)
        assert_almost_equal(pdf1, pdf2)

    def test_halfnorm(self):
        points = [1, 2, 3]
        pdf1 = stats.halfgennorm.pdf(points, 2)
        pdf2 = stats.halfnorm.pdf(points, scale=2 ** (-0.5))
        assert_almost_equal(pdf1, pdf2)

    def test_gennorm(self):
        points = [1, 2, 3]
        pdf1 = stats.halfgennorm.pdf(points, 0.497324)
        pdf2 = stats.gennorm.pdf(points, 0.497324)
        assert_almost_equal(pdf1, 2 * pdf2)