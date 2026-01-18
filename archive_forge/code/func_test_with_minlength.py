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
def test_with_minlength(self):
    x = np.array([0, 1, 0, 1, 1])
    y = np.bincount(x, minlength=3)
    assert_array_equal(y, np.array([2, 3, 0]))
    x = []
    y = np.bincount(x, minlength=0)
    assert_array_equal(y, np.array([]))