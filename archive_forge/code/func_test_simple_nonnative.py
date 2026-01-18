import sys
import warnings
import itertools
import platform
import pytest
import math
from decimal import Decimal
import numpy as np
from numpy.core import umath
from numpy.random import rand, randint, randn
from numpy.testing import (
from numpy.core._rational_tests import rational
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hynp
def test_simple_nonnative(self):
    a = self._generate_non_native_data(self.nr, self.nc)
    m = -0.5
    M = 0.6
    ac = self.fastclip(a, m, M)
    act = self.clip(a, m, M)
    assert_array_equal(ac, act)
    a = self._generate_data(self.nr, self.nc)
    m = -0.5
    M = self._neg_byteorder(0.6)
    assert_(not M.dtype.isnative)
    ac = self.fastclip(a, m, M)
    act = self.clip(a, m, M)
    assert_array_equal(ac, act)