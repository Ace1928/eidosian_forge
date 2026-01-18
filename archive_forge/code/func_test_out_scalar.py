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
def test_out_scalar(self):
    d = np.arange(10)
    out = np.array(0.0)
    r = np.std(d, out=out)
    assert_(r is out)
    assert_array_equal(r, out)
    r = np.var(d, out=out)
    assert_(r is out)
    assert_array_equal(r, out)
    r = np.mean(d, out=out)
    assert_(r is out)
    assert_array_equal(r, out)