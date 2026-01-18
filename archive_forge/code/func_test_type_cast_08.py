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
def test_type_cast_08(self):
    a = self._generate_data(self.nr, self.nc)
    m = -0.5
    M = 1.0
    a_s = self._neg_byteorder(a)
    assert_(not a_s.dtype.isnative)
    ac = self.fastclip(a_s, m, M)
    act = a_s.clip(m, M)
    assert_array_strict_equal(ac, act)