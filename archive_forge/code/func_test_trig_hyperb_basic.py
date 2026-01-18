from mpmath.libmp import *
from mpmath import *
import random
import time
import math
import cmath
def test_trig_hyperb_basic():
    for x in list(range(100)) + list(range(-100, 0)):
        t = x / 4.1
        assert cos(mpf(t)).ae(math.cos(t))
        assert sin(mpf(t)).ae(math.sin(t))
        assert tan(mpf(t)).ae(math.tan(t))
        assert cosh(mpf(t)).ae(math.cosh(t))
        assert sinh(mpf(t)).ae(math.sinh(t))
        assert tanh(mpf(t)).ae(math.tanh(t))
    assert sin(1 + 1j).ae(cmath.sin(1 + 1j))
    assert sin(-4 - 3.6j).ae(cmath.sin(-4 - 3.6j))
    assert cos(1 + 1j).ae(cmath.cos(1 + 1j))
    assert cos(-4 - 3.6j).ae(cmath.cos(-4 - 3.6j))