from __future__ import division
from __future__ import print_function
from builtins import str
from builtins import zip
from builtins import map
from builtins import range
import copy
import weakref
import math
from math import isnan, isinf
import random
import sys
import uncertainties.core as uncert_core
from uncertainties.core import ufloat, AffineScalarFunc, ufloat_fromstr
from uncertainties import umath
def test_numpy_comparison():
    """Comparison with a NumPy array."""
    x = ufloat(1, 0.1)
    assert x != [x, x]
    assert len(x == numpy.arange(10)) == 10
    assert len(numpy.arange(10) == x) == 10
    assert len(x != numpy.arange(10)) == 10
    assert len(numpy.arange(10) != x) == 10
    assert len(x == numpy.array([x, x, x])) == 3
    assert len(numpy.array([x, x, x]) == x) == 3
    assert numpy.all(x == numpy.array([x, x, x]))
    assert len(x < numpy.arange(10)) == 10
    assert len(numpy.arange(10) > x) == 10
    assert len(x <= numpy.arange(10)) == 10
    assert len(numpy.arange(10) >= x) == 10
    assert len(x > numpy.arange(10)) == 10
    assert len(numpy.arange(10) < x) == 10
    assert len(x >= numpy.arange(10)) == 10
    assert len(numpy.arange(10) <= x) == 10
    assert numpy.all((x >= numpy.arange(3)) == [True, False, False])