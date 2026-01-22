import threading
import pickle
import pytest
from copy import deepcopy
import platform
import sys
import math
import numpy as np
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy.stats.sampling import (
from pytest import raises as assert_raises
from scipy import stats
from scipy import special
from scipy.stats import chisquare, cramervonmises
from scipy.stats._distr_params import distdiscrete, distcont
from scipy._lib._util import check_random_state
class dist0:

    def pdf(self, x):
        return 0.5 * (1.0 + np.sin(2.0 * np.pi * x))

    def dpdf(self, x):
        return np.pi * np.cos(2.0 * np.pi * x)

    def cdf(self, x):
        return (1.0 + 2.0 * np.pi * (1 + x) - np.cos(2.0 * np.pi * x)) / (4.0 * np.pi)

    def support(self):
        return (-1, 1)