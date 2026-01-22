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
class TestAdd_newdoc_ufunc:

    def test_ufunc_arg(self):
        assert_raises(TypeError, add_newdoc_ufunc, 2, 'blah')
        assert_raises(ValueError, add_newdoc_ufunc, np.add, 'blah')

    def test_string_arg(self):
        assert_raises(TypeError, add_newdoc_ufunc, np.add, 3)