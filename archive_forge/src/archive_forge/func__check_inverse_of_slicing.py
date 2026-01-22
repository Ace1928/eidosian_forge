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
def _check_inverse_of_slicing(self, indices):
    a_del = delete(self.a, indices)
    nd_a_del = delete(self.nd_a, indices, axis=1)
    msg = 'Delete failed for obj: %r' % indices
    assert_array_equal(setxor1d(a_del, self.a[indices,]), self.a, err_msg=msg)
    xor = setxor1d(nd_a_del[0, :, 0], self.nd_a[0, indices, 0])
    assert_array_equal(xor, self.nd_a[0, :, 0], err_msg=msg)