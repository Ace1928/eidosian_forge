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
@pytest.mark.xfail
@pytest.mark.parametrize('x,alpha,beta,expected', [(0.0001, 1.7720732804618808, 0.5059373136902996, 0.27892916534067), (0.0001, 1.9217001522410235, -0.8779442746685926, 0.281056564327953), (0.0001, 1.5654806051633634, -0.4016220341911392, 0.271252432161167), (0.0001, 1.7420803447784388, -0.38180029468259247, 0.280205311264134), (0.0001, 1.5748002527689913, -0.25200194914153684, 0.280140965235426), (-0.0001, 1.7720732804618808, 0.5059373136902996, 0.278936106741754), (-0.0001, 1.9217001522410235, -0.8779442746685926, 0.281052948629429), (-0.0001, 1.5654806051633634, -0.4016220341911392, 0.271275394392385), (-0.0001, 1.7420803447784388, -0.38180029468259247, 0.280199085645099), (-0.0001, 1.5748002527689913, -0.25200194914153684, 0.280132185432842)])
def test_x_near_zeta(self, x, alpha, beta, expected):
    """Test pdf for x near zeta.

        With S1 parametrization: x0 = x + zeta if alpha != 1 So, for x = 0, x0
        will be close to zeta.

        When case "x near zeta" is not handled properly and quad_eps is not
        low enough: - pdf may be less than 0 - logpdf is nan

        The points from the parametrize block are found randomly so that PDF is
        less than 0.

        Reference values taken from MATLAB
        https://www.mathworks.com/help/stats/stable-distribution.html
        """
    stats.levy_stable.quad_eps = 1.2e-11
    assert_almost_equal(stats.levy_stable.pdf(x, alpha=alpha, beta=beta), expected)