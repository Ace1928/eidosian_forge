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
def test_return_dtype(self):
    assert_equal(select(self.conditions, self.choices, 1j).dtype, np.complex_)
    choices = [choice.astype(np.int8) for choice in self.choices]
    assert_equal(select(self.conditions, choices).dtype, np.int8)
    d = np.array([1, 2, 3, np.nan, 5, 7])
    m = np.isnan(d)
    assert_equal(select([m], [d]), [0, 0, 0, np.nan, 0, 0])