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
def test_type_cast_11(self):
    a = self._generate_non_native_data(self.nr, self.nc)
    b = a.copy()
    b = b.astype(b.dtype.newbyteorder('>'))
    bt = b.copy()
    m = -0.5
    M = 1.0
    self.fastclip(a, m, M, out=b)
    self.clip(a, m, M, out=bt)
    assert_array_strict_equal(b, bt)