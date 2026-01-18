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
def test_return_class_is_ndarray(self):

    class Foo(np.ndarray):

        def __new__(cls, *args, **kwargs):
            return np.array(*args, **kwargs).view(cls)
    a = Foo([1])
    assert_(type(np.allclose(a, a)) is bool)