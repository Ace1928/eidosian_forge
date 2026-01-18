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
def test_signature_two_outputs(self):
    f = vectorize(lambda x: (x, x), signature='()->(),()')
    r = f([1, 2, 3])
    assert_(isinstance(r, tuple) and len(r) == 2)
    assert_array_equal(r[0], [1, 2, 3])
    assert_array_equal(r[1], [1, 2, 3])