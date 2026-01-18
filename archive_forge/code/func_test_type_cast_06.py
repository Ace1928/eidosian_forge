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
def test_type_cast_06(self):
    a = self._generate_data(self.nr, self.nc)
    m = 0.5
    m_s = self._neg_byteorder(m)
    M = 1.0
    act = self.clip(a, m_s, M)
    ac = self.fastclip(a, m_s, M)
    assert_array_strict_equal(ac, act)