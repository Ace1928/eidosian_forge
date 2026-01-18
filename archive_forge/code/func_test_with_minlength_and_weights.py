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
def test_with_minlength_and_weights(self):
    x = np.array([1, 2, 4, 5, 2])
    w = np.array([0.2, 0.3, 0.5, 0.1, 0.2])
    y = np.bincount(x, w, 8)
    assert_array_equal(y, np.array([0, 0.2, 0.5, 0, 0.5, 0.1, 0, 0]))