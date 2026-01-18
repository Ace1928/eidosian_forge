import mpmath
from mpmath import *
from mpmath.libmp import *
import random
import sys
def test_odd_int_bug():
    assert to_int(from_int(3), round_nearest) == 3