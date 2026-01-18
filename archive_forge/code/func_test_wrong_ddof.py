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
def test_wrong_ddof(self):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always', RuntimeWarning)
        assert_array_equal(cov(self.x1, ddof=5), np.array([[np.inf, -np.inf], [-np.inf, np.inf]]))