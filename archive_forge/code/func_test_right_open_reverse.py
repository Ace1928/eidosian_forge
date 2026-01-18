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
def test_right_open_reverse(self):
    x = np.arange(5, -6, -1)
    bins = np.arange(4, -6, -1)
    assert_array_equal(digitize(x, bins, True), np.arange(11))