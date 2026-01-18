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
def test_very_small_input_values(self):
    x = [0.004434375, 0.004756007, 0.003911996, 0.0038005, 0.003409971]
    y = [2.48e-188, 7.41e-181, 4.09e-208, 2.08e-223, 2.66e-245]
    r, p = stats.pearsonr(x, y)
    assert_allclose(r, 0.727293054075045)
    assert_allclose(p, 0.1637805429533202)