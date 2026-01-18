import operator
import warnings
import sys
import decimal
from fractions import Fraction
import math
import pytest
import hypothesis
from hypothesis.extra.numpy import arrays
import hypothesis.strategies as st
from functools import partial
import numpy as np
from numpy import ma
from numpy.testing import (
import numpy.lib.function_base as nfb
from numpy.random import rand
from numpy.lib import (
from numpy.core.numeric import normalize_axis_tuple
def test_out_nan(self):
    with warnings.catch_warnings(record=True):
        warnings.filterwarnings('always', '', RuntimeWarning)
        o = np.zeros((4,))
        d = np.ones((3, 4))
        d[2, 1] = np.nan
        assert_equal(np.median(d, 0, out=o), o)
        o = np.zeros((3,))
        assert_equal(np.median(d, 1, out=o), o)
        o = np.zeros(())
        assert_equal(np.median(d, out=o), o)