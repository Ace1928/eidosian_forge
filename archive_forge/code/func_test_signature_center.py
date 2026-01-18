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
def test_signature_center(self):

    def center(a):
        return a - a.mean()
    f = vectorize(center, signature='(n)->(n)')
    r = f([[1, 3], [2, 4]])
    assert_array_equal(r, [[-1, 1], [-1, 1]])