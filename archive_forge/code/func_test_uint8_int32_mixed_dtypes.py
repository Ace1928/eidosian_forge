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
def test_uint8_int32_mixed_dtypes(self):
    u = np.array([[195, 8, 9]], np.uint8)
    v = np.array([250, 166, 68], np.int32)
    z = np.array([[950, 11010, -30370]], dtype=np.int32)
    assert_equal(np.cross(v, u), z)
    assert_equal(np.cross(u, v), -z)